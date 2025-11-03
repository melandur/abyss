from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from monai.data.meta_tensor import MetaTensor
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    first,
    look_up_option,
    pytorch_after,
)

_nearest_mode = "nearest-exact" if pytorch_after(1, 11) else "nearest"


def sliding_window_inference(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    predictor: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
    overlap: Sequence[float] | float = 0.5,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    """Sliding window inference adopted from MONAI."""

    sw_batch_size = 1
    num_spatial_dims = len(inputs.shape) - 2
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)
    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    roi_size = fall_back_tuple(roi_size, image_size_)

    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(PytorchPadMode.CONSTANT, PytorchPadMode), value=0.0)

    # Store all slices
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = _dense_patch_slices(image_size, roi_size, scan_interval, return_slice=True)

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range: Iterable
    windows_range = range(0, total_slices, sw_batch_size)

    # Create window-level importance map
    valid_patch_size = _get_valid_patch_size(image_size, roi_size)
    try:
        valid_p_size = ensure_tuple(valid_patch_size)
        importance_map_ = _compute_importance_map(
            valid_p_size, mode=mode, sigma_scale=sigma_scale, device='cpu', dtype=compute_dtype
        )
        if len(importance_map_.shape) == num_spatial_dims:
            importance_map_ = importance_map_[None, None]  # adds batch, channel dimensions
    except Exception as e:
        raise RuntimeError(
            f"patch size {valid_p_size}, mode={mode}, sigma_scale={sigma_scale}, device=cpu\n"
            "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
        ) from e
    importance_map_ = convert_data_type(importance_map_, torch.Tensor, device='cpu', dtype=compute_dtype)[0]

    # stores output and count map
    output_image_list, count_map_list, sw_device_buffer, b_s, b_i = [], [], [], 0, 0  # type: ignore
    # for each patch
    for slice_g in windows_range:
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(idx // num_win, idx // num_win + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]

        if sw_batch_size > 1:
            win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice])
        else:
            win_data = inputs[unravel_slice[0]]

        seg_prob_out = predictor(win_data.to('cuda')).to('cpu').detach()  # batched patch

        # convert seg_prob_out to tuple seg_tuple, this does not allocate new memory.
        dict_keys, seg_tuple = _flatten_struct(seg_prob_out)

        w_t = importance_map_
        if len(w_t.shape) == num_spatial_dims:
            w_t = w_t[None, None]
        w_t = w_t.to(dtype=compute_dtype, device='cpu')
        sw_device_buffer = list(seg_tuple)

        for ss in range(len(sw_device_buffer)):
            b_shape = sw_device_buffer[ss].shape
            seg_chns, seg_shape = b_shape[1], b_shape[2:]
            z_scale = None
            if seg_shape != roi_size:
                z_scale = [out_w_i / float(in_w_i) for out_w_i, in_w_i in zip(seg_shape, roi_size)]
                w_t = F.interpolate(w_t, seg_shape, mode=_nearest_mode)
            if len(output_image_list) <= ss:
                output_shape = [batch_size, seg_chns]
                output_shape += [int(_i * _z) for _i, _z in zip(image_size, z_scale)] if z_scale else list(image_size)
                # allocate memory to store the full output and the count for overlapping parts
                new_tensor: Callable = torch.zeros
                output_image_list.append(new_tensor(output_shape, dtype=compute_dtype, device='cpu'))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device='cpu'))
                w_t_ = w_t
                for __s in slices:
                    if z_scale is not None:
                        __s = tuple(slice(int(_si.start * z_s), int(_si.stop * z_s)) for _si, z_s in zip(__s, z_scale))
                    count_map_list[-1][(slice(None), slice(None), *__s)] += w_t_

            sw_device_buffer[ss] *= w_t
            _compute_coords(unravel_slice, z_scale, output_image_list[ss], sw_device_buffer[ss])

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] /= count_map_list.pop(0)

    # remove padding if image_size smaller than roi_size
    if any(pad_size):
        for ss, output_i in enumerate(output_image_list):
            zoom_scale = [_shape_d / _roi_size_d for _shape_d, _roi_size_d in zip(output_i.shape[2:], roi_size)]
            final_slicing: list[slice] = []
            for sp in range(num_spatial_dims):
                si = num_spatial_dims - sp - 1
                slice_dim = slice(
                    int(round(pad_size[sp * 2] * zoom_scale[si])),
                    int(round((pad_size[sp * 2] + image_size_[si]) * zoom_scale[si])),
                )
                final_slicing.insert(0, slice_dim)
            output_image_list[ss] = output_i[(slice(None), slice(None), *final_slicing)]

    final_output = _pack_struct(output_image_list, dict_keys)
    if temp_meta is not None:
        final_output = convert_to_dst_type(final_output, temp_meta, device='cpu')[0]
    else:
        final_output = convert_to_dst_type(final_output, inputs, device='cpu')[0]

    return final_output  # type: ignore


def _dense_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    scan_interval: Sequence[int],
    return_slice: bool = True,
) -> list[tuple[slice, ...]]:
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size: dimensions of image to iterate over
        patch_size: size of patches to generate slices
        scan_interval: dense patch sampling interval
        return_slice: whether to return a list of slices (or tuples of indices), defaults to True

    Returns:
        a list of slice objects defining each patch

    """
    num_spatial_dims = len(image_size)
    patch_size = _get_valid_patch_size(image_size, patch_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i])
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)
    out = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
    if return_slice:
        return [tuple(slice(s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]
    return [tuple((s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]  # type: ignore


def _get_valid_patch_size(image_size: Sequence[int], patch_size: Sequence[int] | int | np.ndarray) -> tuple[int, ...]:
    """
    Given an image of dimensions `image_size`, return a patch size tuple taking the dimension from `patch_size` if this is
    not 0/None. Otherwise, or if `patch_size` is shorter than `image_size`, the dimension from `image_size` is taken. This ensures
    the returned patch size is within the bounds of `image_size`. If `patch_size` is a single number this is interpreted as a
    patch of the same dimensionality of `image_size` with that size in each dimension.
    """
    ndim = len(image_size)
    patch_size_ = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size_))


def _compute_importance_map(
    patch_size: tuple[int, ...],
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    device: torch.device | int | str = "cpu",
    dtype: torch.dtype | str | None = torch.float32,
) -> torch.Tensor:
    """Get importance map for different weight modes.

    Args:
        patch_size: Size of the required importance map. This should be either H, W [,D].
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: Sigma_scale to calculate sigma for each dimension
            (sigma = sigma_scale * dim_size). Used for gaussian mode only.
        device: Device to put importance map on.
        dtype: Data type of the output importance map.

    Raises:
        ValueError: When ``mode`` is not one of ["constant", "gaussian"].

    Returns:
        Tensor of size patch_size.

    """
    mode = look_up_option(mode, BlendMode)
    if mode == BlendMode.CONSTANT:
        importance_map = torch.ones(patch_size, device='cpu', dtype=torch.float)
    elif mode == BlendMode.GAUSSIAN:
        sigma_scale = ensure_tuple_rep(sigma_scale, len(patch_size))
        sigmas = [i * sigma_s for i, sigma_s in zip(patch_size, sigma_scale)]

        for i in range(len(patch_size)):
            x = torch.arange(
                start=-(patch_size[i] - 1) / 2.0, end=(patch_size[i] - 1) / 2.0 + 1, dtype=torch.float, device='cpu'
            )
            x = torch.exp(x**2 / (-2 * sigmas[i] ** 2))  # 1D gaussian
            importance_map = importance_map.unsqueeze(-1) * x[(None,) * i] if i > 0 else x
    else:
        raise ValueError(
            f"Unsupported mode: {mode}, available options are [{BlendMode.CONSTANT}, {BlendMode.CONSTANT}]."
        )
    # handle non-positive weights
    min_non_zero = max(torch.min(importance_map).item(), 1e-3)
    importance_map = torch.clamp_(importance_map.to(torch.float), min=min_non_zero).to(dtype)
    return importance_map


def _compute_coords(coords, z_scale, out, patch):
    """sliding window batch spatial scaling indexing for multi-resolution outputs."""
    for original_idx, p in zip(coords, patch):
        idx_zm = list(original_idx)  # 4D for 2D image, 5D for 3D image
        if z_scale:
            for axis in range(2, len(idx_zm)):
                idx_zm[axis] = slice(
                    int(original_idx[axis].start * z_scale[axis - 2]), int(original_idx[axis].stop * z_scale[axis - 2])
                )
        out[idx_zm] += p


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: Sequence[float]
) -> tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError(f"len(image_size) {len(image_size)} different from spatial dims {num_spatial_dims}.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError(f"len(roi_size) {len(roi_size)} different from spatial dims {num_spatial_dims}.")

    scan_interval = []
    for i, o in zip(range(num_spatial_dims), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def _flatten_struct(seg_out):
    dict_keys = None
    seg_probs: tuple[torch.Tensor, ...]
    if isinstance(seg_out, torch.Tensor):
        seg_probs = (seg_out,)
    elif isinstance(seg_out, Mapping):
        dict_keys = sorted(seg_out.keys())  # track predictor's output keys
        seg_probs = tuple(seg_out[k] for k in dict_keys)
    else:
        seg_probs = ensure_tuple(seg_out)
    return dict_keys, seg_probs


def _pack_struct(seg_out, dict_keys=None):
    if dict_keys is not None:
        return dict(zip(dict_keys, seg_out))
    if isinstance(seg_out, (list, tuple)) and len(seg_out) == 1:
        return seg_out[0]
    return ensure_tuple(seg_out)
