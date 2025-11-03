import os
from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from torch import nn

torch._dynamo.config.cache_size_limit = 24  # increase cache size limit

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


def get_kernels_strides(config):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.
    """
    sizes, spacings = config['trainer']['patch_size'], config['dataset']['spacing']

    for size in sizes:
        if size % 8 != 0:
            raise ValueError(f"Patch size is not supported, please try to modify the size {size} to be divisible by 8.")

    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])

    results = {
        'kernel_size': kernels,
        'strides': strides,
        'upsample_kernel_size': strides[1:],
    }
    return results


def get_network(config):
    dimensions = get_kernels_strides(config)
    in_channels = len(config['dataset']['channel_order'])
    out_channels = len(config['trainer']['label_classes'])

    net = ResidualEncoderUNet(
        input_channels=in_channels,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=torch.nn.modules.conv.Conv3d,
        kernel_sizes=dimensions['kernel_size'],
        strides=dimensions['strides'],
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=out_channels,
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        conv_bias=True,
        norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-05, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=True,
    )

    # Load pretrained weights if checkpoint path is provided
    pretrained_checkpoint_path = config['training'].get('pretrained_checkpoint_path')
    if pretrained_checkpoint_path is not None and os.path.exists(pretrained_checkpoint_path):
        logger.info(f'Loading pretrained weights from: {pretrained_checkpoint_path}')
        net = load_pretrained_weights(
            net,
            pretrained_weights_path=pretrained_checkpoint_path,
            downstream_input_channels=in_channels,
            downstream_input_patchsize=config['trainer']['patch_size'],
        )[0]
    elif pretrained_checkpoint_path is not None:
        logger.warning(f'Pretrained checkpoint not found: {pretrained_checkpoint_path}')
        logger.info('Continuing with randomly initialized weights')
    else:
        logger.info('No pretrained checkpoint specified, using randomly initialized weights')

    if config['training']['compile']:
        net = torch.compile(net)

    return net


def load_pretrained_weights(
    network,
    pretrained_weights_path: str,
    downstream_input_channels: int,
    downstream_input_patchsize: Union[int, list[int]],
    pt_input_channels: Union[int, None] = None,
    pt_input_patchsize: Union[int, list[int], None] = None,
    pt_key_to_encoder: Union[str, None] = None,
    pt_key_to_stem: Union[str, None] = None,
    pt_keys_to_in_proj: Union[tuple[str, ...], None] = None,
    pt_key_to_lpe: Union[str, None] = None,
) -> tuple[nn.Module, bool]:
    """
    Load pretrained weights from nnssl checkpoint into the network.

    Following nnUNet v2 / TaWald approach: All parameters are automatically extracted
    from the checkpoint's 'nnssl_adaptation_plan'. Function arguments are only used
    as fallbacks if not present in the checkpoint.

    Per default we only load the encoder and the stem weights. The stem weights are
    adapted to the number of input channels through repeats. The decoder is initialized from scratch.

    Args:
        network: The neural network to load weights into.
        pretrained_weights_path: Path to the pretrained weights file (.pth).
        downstream_input_channels: Number of input channels for downstream task.
        downstream_input_patchsize: Patch size for downstream task (int or list).
        pt_input_channels: Pretrained input channels (optional, extracted from checkpoint if available).
        pt_input_patchsize: Pretrained patch size (optional, extracted from checkpoint if available).
        pt_key_to_encoder: Key to encoder in pretrained model (optional, extracted from checkpoint if available).
        pt_key_to_stem: Key to stem in pretrained model (optional, extracted from checkpoint if available).
        pt_keys_to_in_proj: Keys to input projection layers (optional, extracted from checkpoint if available).
        pt_key_to_lpe: Key to learnable positional embedding (optional, extracted from checkpoint if available).

    Returns:
        Tuple of (network with loaded weights, boolean indicating if channel mismatch occurred).
    """
    # Validate checkpoint file exists
    if not os.path.exists(pretrained_weights_path):
        raise FileNotFoundError(f'Pretrained checkpoint not found: {pretrained_weights_path}')

    logger.info(f'Loading checkpoint: {pretrained_weights_path}')

    # Load checkpoint
    try:
        ckp = torch.load(pretrained_weights_path, weights_only=True, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f'Failed to load checkpoint: {e}') from e

    # Validate checkpoint structure
    if 'network_weights' not in ckp:
        raise ValueError(f'Checkpoint missing "network_weights" key: {pretrained_weights_path}')
    if 'nnssl_adaptation_plan' not in ckp:
        raise ValueError(
            f'Checkpoint missing "nnssl_adaptation_plan" key (not an nnssl checkpoint): {pretrained_weights_path}'
        )

    pre_train_statedict: dict[str, torch.Tensor] = ckp['network_weights']
    adaptation_plan = ckp['nnssl_adaptation_plan']

    logger.debug(f'Checkpoint adaptation plan keys: {list(adaptation_plan.keys())}')

    # Extract parameters from nnssl_adaptation_plan (priority over function arguments)
    # Following TaWald/nnUNet v2 approach: checkpoint is source of truth
    pt_key_to_stem = adaptation_plan.get('key_to_stem', pt_key_to_stem)
    pt_key_to_encoder = adaptation_plan.get('key_to_encoder', pt_key_to_encoder)
    pt_keys_to_in_proj = adaptation_plan.get('keys_to_in_proj', pt_keys_to_in_proj)
    pt_key_to_lpe = adaptation_plan.get('key_to_lpe', pt_key_to_lpe)
    pt_input_patchsize = adaptation_plan.get('pretrain_patch_size', pt_input_patchsize)
    pt_input_channels = adaptation_plan.get('pretrain_input_channels', pt_input_channels)

    # Validate required parameters are available
    if pt_key_to_stem is None:
        raise ValueError('key_to_stem not found in checkpoint and not provided')
    if pt_key_to_encoder is None:
        raise ValueError('key_to_encoder not found in checkpoint and not provided')
    if pt_keys_to_in_proj is None:
        raise ValueError('keys_to_in_proj not found in checkpoint and not provided')
    if pt_input_channels is None:
        raise ValueError('pretrain_input_channels not found in checkpoint and not provided')
    if pt_input_patchsize is None:
        raise ValueError('pretrain_patch_size not found in checkpoint and not provided')

    # Convert patch sizes to lists if needed
    if isinstance(downstream_input_patchsize, int):
        downstream_input_patchsize = [downstream_input_patchsize] * 3
    if isinstance(pt_input_patchsize, int):
        pt_input_patchsize = [pt_input_patchsize] * 3

    logger.info(f'Pretrained model: {pt_input_channels} channels, patch_size={pt_input_patchsize}')
    logger.info(f'Downstream model: {downstream_input_channels} channels, patch_size={downstream_input_patchsize}')
    logger.debug(f'Encoder key: {pt_key_to_encoder}, Stem key: {pt_key_to_stem}')

    # Get network architecture keys
    key_to_encoder = network.key_to_encoder
    key_to_stem = network.key_to_stem

    stem_in_encoder = pt_key_to_stem in pre_train_statedict
    pt_weight_in_ch_mismatch = False
    need_to_adapt_lpe = False
    key_to_lpe = getattr(network, 'key_to_lpe', None)
    lpe_in_stem = False
    lpe_in_encoder = False

    # Check if network uses learnable positional embedding
    if key_to_lpe is not None:
        try:
            network.get_submodule(key_to_lpe)
        except AttributeError:
            key_to_lpe = None

    if key_to_lpe is not None:
        lpe_in_encoder = key_to_lpe.startswith(key_to_encoder)
        lpe_in_stem = key_to_lpe.startswith(key_to_stem)

    # Check if patch size adaptation is needed for LPE
    if pt_input_patchsize != downstream_input_patchsize:
        need_to_adapt_lpe = True
        logger.debug('Patch size mismatch detected, LPE adaptation needed')

    def strip_dot_prefix(s: str) -> str:
        """Strip dot prefix from module keys."""
        if s.startswith('.'):
            return s[1:]
        return s

    # ----- Match the keys of pretrained weights to the current architecture ----- #
    if stem_in_encoder:
        encoder_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_encoder)}
        if downstream_input_channels > pt_input_channels:
            pt_weight_in_ch_mismatch = True
            logger.info(f'Adapting stem weights: {pt_input_channels} -> {downstream_input_channels} channels')
            k_proj = pt_keys_to_in_proj[0] + '.weight'
            vals = (encoder_weights[k_proj].repeat(1, downstream_input_channels, 1, 1)) / downstream_input_channels
            for k in pt_keys_to_in_proj:
                encoder_weights[k] = vals
        # Fix the path to the weights:
        new_encoder_weights = {
            strip_dot_prefix(k.replace(pt_key_to_encoder, "")): v for k, v in encoder_weights.items()
        }
        # --------------------------------- Adapt LPE -------------------------------- #
        if need_to_adapt_lpe:
            if lpe_in_encoder:
                handle_pos_embed_resize(
                    new_encoder_weights,
                    network.get_submodule(key_to_encoder).state_dict(),
                    'interpolate_trilinear',
                    downstream_input_patchsize,
                    pt_input_patchsize,
                    new_encoder_weights['down_projection.proj.weight'].shape[2:],
                )
                new_encoder_weights['pos_embed'].to(next(network.parameters()).device)
            if 'cls_token' in encoder_weights.keys():
                skip_strings_in_pretrained = ['cls_token']
                new_encoder_weights, _ = filter_state_dict(encoder_weights, skip_strings_in_pretrained)

        # Load encoder weights
        encoder_module = network.get_submodule(key_to_encoder)
        encoder_module.load_state_dict(new_encoder_weights, strict=False)
        logger.success('Encoder weights loaded successfully')
    else:
        encoder_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_encoder)}
        stem_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_stem)}
        if downstream_input_channels > pt_input_channels:
            pt_weight_in_ch_mismatch = True
            logger.info(f'Adapting stem weights: {pt_input_channels} -> {downstream_input_channels} channels')
            k_proj = pt_keys_to_in_proj[0] + '.weight'
            vals = (stem_weights[k_proj].repeat(1, downstream_input_channels, 1, 1, 1)) / downstream_input_channels
            for k in pt_keys_to_in_proj:
                stem_weights[k + '.weight'] = vals
        new_encoder_weights = {
            strip_dot_prefix(k.replace(pt_key_to_encoder, "")): v for k, v in encoder_weights.items()
        }
        new_stem_weights = {strip_dot_prefix(k.replace(pt_key_to_stem, "")): v for k, v in stem_weights.items()}
        # --------------------------------- Adapt LPE -------------------------------- #
        if need_to_adapt_lpe:
            if lpe_in_stem:  # Since stem not in encoder we need to take care of lpe in it here
                handle_pos_embed_resize(
                    new_stem_weights,
                    network.get_submodule(key_to_stem).state_dict(),
                    'interpolate_trilinear',
                    downstream_input_patchsize,
                    pt_input_patchsize,
                    new_stem_weights['proj.weight'].shape[2:],
                )
                new_stem_weights['pos_embed'].to(next(network.parameters()).device)
            elif lpe_in_encoder:
                handle_pos_embed_resize(
                    new_encoder_weights,
                    network.get_submodule(key_to_encoder).state_dict(),
                    'interpolate_trilinear',
                    downstream_input_patchsize,
                    pt_input_patchsize,
                    new_stem_weights['proj.weight'].shape[2:],
                )
                new_encoder_weights['pos_embed'].to(next(network.parameters()).device)

        if 'cls_token' in encoder_weights.keys():
            skip_strings_in_pretrained = ['cls_token']
            new_encoder_weights, _ = filter_state_dict(encoder_weights, skip_strings_in_pretrained)

        # Load encoder and stem weights
        encoder_module = network.get_submodule(key_to_encoder)
        encoder_module.load_state_dict(new_encoder_weights, strict=False)
        stem_module = network.get_submodule(key_to_stem)
        stem_module.load_state_dict(new_stem_weights, strict=False)
        logger.success('Encoder and stem weights loaded successfully')

    if not need_to_adapt_lpe and key_to_lpe is not None:
        # Load the positional embedding weights
        lpe_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_lpe)}
        if len(lpe_weights) != 1:
            raise ValueError(
                f'Found multiple lpe weights, but expect only a single tensor. Got {list(lpe_weights.keys())}'
            )
        network.get_parameter(key_to_lpe).data = list(lpe_weights.values())[0].to(next(network.parameters()).device)
        logger.debug('Learnable positional embedding loaded')

    logger.success('Pretrained weights loaded successfully')

    # Cleanup
    if 'encoder_weights' in locals():
        del pre_train_statedict, encoder_weights
    if 'new_encoder_weights' in locals():
        del new_encoder_weights

    return network, pt_weight_in_ch_mismatch


def filter_state_dict(state_dict, skip_strings):
    found_flag = False
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if any(skip in k for skip in skip_strings):
            found_flag = True
            continue
        filtered_state_dict[k] = v

    return filtered_state_dict, found_flag


def interpolate_patch_embed_1d(patch_embed, target_len, mode="linear"):
    """Resizes patch embeddings using interpolation."""
    return F.interpolate(
        patch_embed.permute(0, 2, 1),  # [B, C, Tokens]
        size=target_len,
        mode=mode,
        align_corners=False,
    ).permute(
        0, 2, 1
    )  # [B, Tokens, C]


def interpolate_patch_embed_3d(patch_embed, in_shape, out_shape):
    """Resizes patch embeddings using 3D trilinear interpolation."""
    patch_embed = patch_embed.permute(0, 2, 1)
    patch_embed = rearrange(patch_embed, "B C (x y z) -> B C x y z", **in_shape)
    patch_embed = F.interpolate(patch_embed, size=list(out_shape.values()), mode="trilinear", align_corners=False)
    patch_embed = rearrange(patch_embed, "B C x y z -> B C (x y z)", **out_shape)
    return patch_embed.permute(0, 2, 1)


def handle_pos_embed_resize(
    pretrained_dict, model_dict, mode, input_shape=None, pretrained_input_patch_size=None, patch_embed_size=None
):
    pretrained_pos_embed = pretrained_dict['pos_embed']
    model_pos_embed = model_dict['pos_embed']
    model_pos_embed_shape = model_pos_embed.shape

    has_cls_token = 'cls_token' in pretrained_dict
    cls_pos_embed = None

    if has_cls_token:
        cls_pos_embed = pretrained_pos_embed[:, :1, :]
        patch_pos_embed = pretrained_pos_embed[:, 1:, :]
    else:
        if 'cls_token' in model_dict.keys():
            cls_pos_embed = model_pos_embed[:, :1, :]
        patch_pos_embed = pretrained_pos_embed

    if mode == 'interpolate':
        resized_patch_pos_embed = interpolate_patch_embed_1d(
            patch_pos_embed, target_len=model_pos_embed_shape[1] - int(has_cls_token)
        )

    elif mode == 'interpolate_trilinear':
        # Calculate input/output 3D shapes
        in_shape = dict(zip('xyz', [int(d / p) for d, p in zip(pretrained_input_patch_size, patch_embed_size)]))
        out_shape = dict(zip('xyz', [int(d / p) for d, p in zip(input_shape, patch_embed_size)]))
        resized_patch_pos_embed = interpolate_patch_embed_3d(patch_pos_embed, in_shape, out_shape)

    else:
        raise NotImplementedError(f'Unknown resize mode: {mode}')

    if cls_pos_embed is not None:
        resized_pos_embed = torch.cat([cls_pos_embed, resized_patch_pos_embed], dim=1)
    else:
        resized_pos_embed = resized_patch_pos_embed
    pretrained_dict['pos_embed'] = resized_pos_embed
