# pylint: disable-all

from monai.networks.nets import UNet, resnet10

unet = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

resnet_10 = resnet10(
    pretrained=False,
    spatial_dims=3,
    n_input_channels=4,
    num_classes=2,
)

#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.ndimage.filters import gaussian_filter


def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=True)
    return data


def flip(x, dim):
    """
    flips the tensor at dimension dim (mirroring!)
    :param x:
    :param dim:
    :return:
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if (
            isinstance(module, nn.Conv3d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.ConvTranspose2d)
            or isinstance(module, nn.ConvTranspose3d)
        ):
            module.weight = nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class ConvDropoutNormNonlin(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv2d,
        conv_kwargs=None,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
    ):

        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs['p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class StackedConvLayers(nn.Module):
    def __init__(
        self,
        input_feature_channels,
        output_feature_channels,
        num_convs,
        conv_op=nn.Conv2d,
        conv_kwargs=None,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        first_stride=None,
    ):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        super().__init__()
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        self.blocks = nn.Sequential(
            *(
                [
                    ConvDropoutNormNonlin(
                        input_feature_channels,
                        output_feature_channels,
                        self.conv_op,
                        self.conv_kwargs_first_conv,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    )
                ]
                + [
                    ConvDropoutNormNonlin(
                        output_feature_channels,
                        output_feature_channels,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    )
                    for _ in range(num_convs - 1)
                ]
            )
        )

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if (
        isinstance(module, nn.Conv2d)
        or isinstance(module, nn.Conv3d)
        or isinstance(module, nn.Dropout3d)
        or isinstance(module, nn.Dropout2d)
        or isinstance(module, nn.Dropout)
        or isinstance(module, nn.InstanceNorm3d)
        or isinstance(module, nn.InstanceNorm2d)
        or isinstance(module, nn.InstanceNorm1d)
        or isinstance(module, nn.BatchNorm2d)
        or isinstance(module, nn.BatchNorm3d)
        or isinstance(module, nn.BatchNorm1d)
    ):
        logger.exception(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return F.interpolate(
            x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )


class Generic_UNet(nn.Module):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(
        self,
        input_channels,
        base_num_features,
        num_classes,
        num_pool,
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=nn.Conv2d,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        deep_supervision=True,
        dropout_in_localization=False,
        final_nonlin=softmax_helper,
        weightInitializer=InitWeights_He(1e-2),
        pool_op_kernel_sizes=None,
        conv_kernel_sizes=None,
        upscale_logits=False,
        convolutional_pooling=False,
        convolutional_upsampling=False,
        max_num_features=None,
    ):

        super().__init__()
        self.input_shape_must_be_divisible_by = None
        self.conv_op = None
        self.num_classes = None
        self.inference_apply_nonlin = lambda x: x

        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self.do_ds = deep_supervision

        upsample_mode = 'trilinear'
        pool_op = nn.MaxPool3d
        transpconv = nn.ConvTranspose3d
        if pool_op_kernel_sizes is None:
            pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            tmp = StackedConvLayers(
                input_features,
                output_features,
                num_conv_per_stage,
                self.conv_op,
                self.conv_kwargs,
                self.norm_op,
                self.norm_op_kwargs,
                self.dropout_op,
                self.dropout_op_kwargs,
                self.nonlin,
                self.nonlin_kwargs,
                first_stride,
            )
            self.conv_blocks_context.append(tmp)
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(
            nn.Sequential(
                StackedConvLayers(
                    input_features,
                    output_features,
                    num_conv_per_stage - 1,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                    first_stride,
                ),
                StackedConvLayers(
                    output_features,
                    final_num_features,
                    1,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                ),
            )
        )

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)
            ].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(
                    transpconv(
                        nfeatures_from_down,
                        nfeatures_from_skip,
                        pool_op_kernel_sizes[-(u + 1)],
                        pool_op_kernel_sizes[-(u + 1)],
                        bias=False,
                    )
                )

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[-(u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[-(u + 1)]
            self.conv_blocks_localization.append(
                nn.Sequential(
                    StackedConvLayers(
                        n_features_after_tu_and_concat,
                        nfeatures_from_skip,
                        num_conv_per_stage - 1,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    ),
                    StackedConvLayers(
                        nfeatures_from_skip,
                        final_num_features,
                        1,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    ),
                )
            )

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(
                conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes, 1, 1, 0, 1, 1, False)
            )

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(
                    Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]), mode=upsample_mode)
                )
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops
            )  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self.do_ds:
            return tuple(
                [seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]
            )
        else:
            return seg_outputs[-1]

    def get_device(self):
        if next(self.parameters()).device == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def set_device(self, device):
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)

    def predict_3D(
        self,
        x,
        do_mirroring: bool,
        num_repeats=1,
        batch_size=1,
        mirror_axes=(0, 1, 2),
        tiled=False,
        tile_in_z=True,
        step=2,
        patch_size=None,
        regions_class_order=None,
        use_gaussian=False,
        pad_border_mode="edge",
        device='cpu',
        half=False,
        pad_kwargs=None,
        all_in_gpu=False,
    ):
        """
        :param x: (c, x, y , z)
        :param do_mirroring: whether or not to do test time data augmentation by mirroring
        :param num_repeats: how often should each patch be predicted? This MUST be 1 unless you are using monte carlo
        dropout sampling (for which you also must set use_train_mode=True)
        :param use_train_mode: sets the model to train mode. This functionality is kinda broken because it should not
        set batch norm to train mode! Do not use!
        :param batch_size: also used for monte carlo sampling, leave it at 1
        :param mirror_axes: the spatial axes along which the mirroring takes place, if applicable
        :param tiled: if False then prediction is fully convolutional (careful with large images!). Else we use sliding window
        :param tile_in_z: what a bad name. If x is (c, x, y, z), then this sets whether we do for sliding window the
        axis x or whether we do that one fully convolutionally. I suggest you don't use this (set tile_in_z=True)
        :param step: how large is the step size for sliding window? 2 = patch_size // 2 for each axis
        :param patch_size: if tiled prediction, how large are the patches that we use?
        :param regions_class_order: Don't use this. Fabian only.
        :param use_gaussian: set this to True to prevent stitching artifacts
        :param all_in_gpu: only affects _internal_predict_3D_3Dconv_tiled, _internal_predict_3D_2Dconv_tiled, _internal_predict_3D_2Dconv,
        _internal_predict_2D_2Dconv_tiled
        :return:
        """
        logger.debug("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if len(mirror_axes) > 0 and max(mirror_axes) > 2:
            logger.exception("mirror axes. duh")

        self.eval()
        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        if self.conv_op == nn.Conv3d:
            if tiled:
                res = self._internal_predict_3D_3Dconv_tiled(
                    x,
                    num_repeats,
                    batch_size,
                    tile_in_z,
                    step,
                    do_mirroring,
                    mirror_axes,
                    patch_size,
                    regions_class_order,
                    use_gaussian,
                    pad_border_mode,
                    pad_kwargs=pad_kwargs,
                    all_in_gpu=all_in_gpu,
                    device=device,
                    half=half,
                )
        else:
            raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")
        return res

    @staticmethod
    def pad_nd_image(
        image,
        new_shape=None,
        mode="constant",
        kwargs=None,
        return_slicer=False,
        shape_must_be_divisible_by=None,
    ):
        """
        one padder to pad them all. Documentation? Well okay. A little bit

        :param image: nd image. can be anything
        :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
        len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
        the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
        Example:
        image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
        image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

        :param mode: see np.pad for documentation
        :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
        to original shape
        :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
        divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
        be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
        :param kwargs: see np.pad for documentation
        """
        if kwargs is None:
            kwargs = {'constant_values': 0}

        if new_shape is not None:
            old_shape = np.array(image.shape[-len(new_shape) :])
        else:
            assert shape_must_be_divisible_by is not None
            assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
            new_shape = image.shape[-len(shape_must_be_divisible_by) :]
            old_shape = new_shape

        num_axes_nopad = len(image.shape) - len(new_shape)

        new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

        if not isinstance(new_shape, np.ndarray):
            new_shape = np.array(new_shape)

        if shape_must_be_divisible_by is not None:
            if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
                shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
            else:
                assert len(shape_must_be_divisible_by) == len(new_shape)

            for i in range(len(new_shape)):
                if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                    new_shape[i] -= shape_must_be_divisible_by[i]

            new_shape = np.array(
                [
                    new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i]
                    for i in range(len(new_shape))
                ]
            )

        difference = new_shape - old_shape
        pad_below = difference // 2
        pad_above = difference // 2 + difference % 2
        pad_list = [[0, 0]] * num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

        if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
            res = np.pad(image, pad_list, mode, **kwargs)
        else:
            res = image

        if not return_slicer:
            return res
        else:
            pad_list = np.array(pad_list)
            pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
            slicer = list(slice(*i) for i in pad_list)
            return res, slicer

    def _internal_predict_3D_3Dconv_tiled(
        self,
        x,
        num_repeats,
        BATCH_SIZE=None,
        tile_in_z=True,
        step=2,
        do_mirroring=True,
        mirror_axes=(0, 1, 2),
        patch_size=None,
        regions_class_order=None,
        use_gaussian=False,
        pad_border_mode="edge",
        pad_kwargs=None,
        all_in_gpu=False,
        device='cpu',
        half=False,
    ):
        """
        x must be (c, x, y, z)
        :param x:
        :param num_repeats:
        :param BATCH_SIZE:
        :param tile_in_z:
        :param step:
        :param do_mirroring:
        :param mirror_axes:
        :param patch_size:
        :param regions_class_order:
        :param use_gaussian:
        :param pad_border_mode:
        :param pad_kwargs:
        :param all_in_gpu: if True then data and prediction will be held in GPU for inference. Faster, but uses more vram
        :return:
        """
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        with torch.no_grad():
            assert patch_size is not None, "patch_size cannot be None for tiled prediction"

            data, slicer = self.pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
            data = data[None]

            if BATCH_SIZE is not None:
                data = np.vstack([data] * BATCH_SIZE)

            nb_of_classes = 4

            if use_gaussian:
                tmp = np.zeros(patch_size)
                center_coords = [i // 2 for i in patch_size]
                sigmas = [i // 8 for i in patch_size]
                tmp[tuple(center_coords)] = 1
                tmp_smooth = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
                tmp_smooth = tmp_smooth / tmp_smooth.max() * 1
                add = tmp_smooth + 1e-8
            else:
                add = np.ones(patch_size, dtype=np.float32)

            add = add.astype(np.float32)

            data_shape = data.shape
            center_coord_start = np.array([i // 2 for i in patch_size]).astype(int)
            center_coord_end = np.array(
                [data_shape[i + 2] - patch_size[i] // 2 for i in range(len(patch_size))]
            ).astype(int)
            num_steps = np.ceil(
                [(center_coord_end[i] - center_coord_start[i]) / (patch_size[i] / step) for i in range(3)]
            )
            step_size = np.array(
                [(center_coord_end[i] - center_coord_start[i]) / (num_steps[i] + 1e-8) for i in range(3)]
            )
            step_size[step_size == 0] = 9999999
            xsteps = np.round(np.arange(center_coord_start[0], center_coord_end[0] + 1e-8, step_size[0])).astype(int)
            ysteps = np.round(np.arange(center_coord_start[1], center_coord_end[1] + 1e-8, step_size[1])).astype(int)
            zsteps = np.round(np.arange(center_coord_start[2], center_coord_end[2] + 1e-8, step_size[2])).astype(int)

            if all_in_gpu:
                # some of these can remain in half. We just need the reuslts for softmax so it won't hurt at all to reduce
                # precision. Inference is of course done in float
                result = torch.zeros([nb_of_classes] + list(data.shape[2:]), dtype=torch.half).cuda()
                data = torch.from_numpy(data).cuda(self.get_device())
                result_numsamples = torch.zeros([nb_of_classes] + list(data.shape[2:]), dtype=torch.half).cuda()
                add = torch.from_numpy(add).cuda(self.get_device()).float()
                add_torch = add
            else:
                result = np.zeros([nb_of_classes] + list(data.shape[2:]), dtype=np.float32)
                result_numsamples = np.zeros([nb_of_classes] + list(data.shape[2:]), dtype=np.float32)

                if device == 'cpu':
                    add_torch = torch.from_numpy(add)
                else:
                    if half:
                        add_torch = torch.from_numpy(add).cuda(self.get_device(), non_blocking=True).half()
                    else:
                        add_torch = torch.from_numpy(add).cuda(self.get_device(), non_blocking=True)

            # data, result and add_torch and result_numsamples are now on GPU
            for x in xsteps:
                lb_x = x - patch_size[0] // 2
                ub_x = x + patch_size[0] // 2
                for y in ysteps:
                    lb_y = y - patch_size[1] // 2
                    ub_y = y + patch_size[1] // 2
                    for z in zsteps:
                        lb_z = z - patch_size[2] // 2
                        ub_z = z + patch_size[2] // 2

                        predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                            data[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z],
                            num_repeats,
                            mirror_axes,
                            do_mirroring,
                            add_torch,
                            device,
                            half,
                        )[0]
                        if all_in_gpu:
                            predicted_patch = predicted_patch.half()
                        else:
                            predicted_patch = predicted_patch.cpu().numpy()

                        result[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch

                        if all_in_gpu:
                            result_numsamples[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add.half()
                        else:
                            result_numsamples[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add

            slicer = tuple(
                [slice(0, result.shape[i]) for i in range(len(result.shape) - (len(slicer) - 1))] + slicer[1:]
            )
            result = result[slicer]
            result_numsamples = result_numsamples[slicer]

            softmax_pred = result / result_numsamples

            # patient_data = patient_data[:, :old_shape[0], :old_shape[1], :old_shape[2]]
            if regions_class_order is None:
                predicted_segmentation = softmax_pred.argmax(0)
            else:
                if all_in_gpu:
                    softmax_pred_here = softmax_pred.detach().cpu().numpy()
                else:
                    softmax_pred_here = softmax_pred
                predicted_segmentation_shp = softmax_pred_here[0].shape
                predicted_segmentation = np.zeros(predicted_segmentation_shp, dtype=np.float32)
                for i, c in enumerate(regions_class_order):
                    predicted_segmentation[softmax_pred_here[i] > 0.5] = c

            if all_in_gpu:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
                softmax_pred = softmax_pred.half().detach().cpu().numpy()
        return predicted_segmentation, None, softmax_pred, None

    def _internal_maybe_mirror_and_pred_3D(
        self,
        x,
        num_repeats,
        mirror_axes,
        do_mirroring=True,
        mult=None,
        device='cpu',
        half=False,
    ):
        # everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        # we now return a cuda tensor! Not numpy array!
        with torch.no_grad():
            if device == 'cpu':
                x = maybe_to_torch(x)
                result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]), dtype=torch.float)
                mult = maybe_to_torch(mult)
            else:
                if half:
                    x = to_cuda(maybe_to_torch(x).half(), gpu_id=self.get_device())
                    result_torch = (
                        torch.zeros([1, self.num_classes] + list(x.shape[2:]), dtype=torch.float)
                        .cuda(self.get_device(), non_blocking=True)
                        .half()
                    )
                    mult = to_cuda(maybe_to_torch(mult).half(), gpu_id=self.get_device())
                else:
                    x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
                    result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]), dtype=torch.float).cuda(
                        self.get_device(), non_blocking=True
                    )
                    mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())

            num_results = num_repeats
            if do_mirroring:
                mirror_idx = 8
                num_results *= 2 ** len(mirror_axes)
            else:
                mirror_idx = 1

            for i in range(num_repeats):
                for m in range(mirror_idx):
                    if m == 0:
                        pred = self.inference_apply_nonlin(self(x))
                        result_torch += 1 / num_results * pred

                    if m == 1 and (2 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(x, 4)))
                        result_torch += 1 / num_results * flip(pred, 4)

                    if m == 2 and (1 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(x, 3)))
                        result_torch += 1 / num_results * flip(pred, 3)

                    if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(flip(x, 4), 3)))
                        result_torch += 1 / num_results * flip(flip(pred, 4), 3)

                    if m == 4 and (0 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(x, 2)))
                        result_torch += 1 / num_results * flip(pred, 2)

                    if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(flip(x, 4), 2)))
                        result_torch += 1 / num_results * flip(flip(pred, 4), 2)

                    if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(flip(x, 3), 2)))
                        result_torch += 1 / num_results * flip(flip(pred, 3), 2)

                    if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(flip(flip(x, 3), 2), 4)))
                        result_torch += 1 / num_results * flip(flip(flip(pred, 3), 2), 4)

            if mult is not None:
                result_torch[:, :] *= mult

        return result_torch


nn_unet = Generic_UNet(
    input_channels=4,
    base_num_features=30,
    num_classes=4,
    num_pool=5,
    num_conv_per_stage=2,
    feat_map_mul_on_downscale=2,
    conv_op=torch.nn.Conv3d,
    norm_op=torch.nn.InstanceNorm3d,
    norm_op_kwargs={'eps': 1e-5, 'affine': True},
    dropout_op=torch.nn.Dropout3d,
    dropout_op_kwargs={'p': 0, 'inplace': True},
    nonlin=torch.nn.LeakyReLU,
    nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
    deep_supervision=False,
    dropout_in_localization=False,
    final_nonlin=lambda x: x,
    weightInitializer=InitWeights_He(1e-2),
    pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    upscale_logits=False,
    convolutional_pooling=True,
    convolutional_upsampling=True,
    max_num_features=None,
)

STORAGE_BASE_PATH = os.path.join(os.path.expanduser('~'), '.deepbratumia', 'storage')
MODELS_STORAGE_PATH = os.path.join(STORAGE_BASE_PATH, 'models')
HD_GLIO_MODELS = os.path.join(MODELS_STORAGE_PATH, 'hd_glio', 'hd_glio_params')

weight_path = os.path.join(HD_GLIO_MODELS, 'fold_0', 'model_best.model')
param = torch.load(weight_path, map_location=lambda storage, loc: storage)['state_dict']
nn_unet.load_state_dict(param, strict=True)
print(nn_unet)
