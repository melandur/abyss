import os
import re

import torch
from grpc.beta.implementations import insecure_channel

from abyss.training.network_definitions import DynUNet
from monai.networks.nets import resnet18

def get_kernels_strides(config):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.
    """
    sizes, spacings = config['trainer']['patch_size'], config['dataset']['spacing']
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

    net = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        filters=[64, 128, 256, 512, 512, 512],  # brats winner 22
        # filters=[64, 96, 128, 192, 256, 384, 512, 768, 1024],  # brats winner 21
        # filters=[30, 60, 120, 240, 320, 320],  # nnunet winner 18/19
        kernel_size=dimensions['kernel_size'],
        strides=dimensions['strides'],
        upsample_kernel_size=dimensions['upsample_kernel_size'],
        dropout=None,
        norm_name=('instance', {'affine': True}),
        act_name=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
        deep_supervision=True,
        deep_supr_num=2,
        res_block=True,
        trans_bias=True,
    )

    # if config['training']['reload_checkpoint']:
    #     ckpt_folder_path = config['project']['results_path']
    #     best_ckpt_path = os.path.join(ckpt_folder_path, 'best.ckpt')
    #     if not os.path.exists(best_ckpt_path):
    #         raise FileNotFoundError(f'checkpoint: {best_ckpt_path} not found')
    #
    #     ckpt = torch.load(best_ckpt_path)
    #     weights = ckpt['state_dict']
    #     for key in list(weights.keys()):
    #         if 'net._orig_mod.' in key:
    #             new_key = key.replace('net._orig_mod.', '')
    #             weights[new_key] = weights.pop(key)
    #         # if 'criterion.' in key:
    #         #     weights.pop(key)
    #
    #     net.load_state_dict(weights)

    # ckpt_folder_path = config['project']['results_path']
    # best_ckpt_path = os.path.join(ckpt_folder_path, 'best.ckpt')
    # ckpt = torch.load(best_ckpt_path)

    # print(ckpt.keys())
    # print(ckpt['optimizer_states'])


    if config['training']['compile']:
        net = torch.compile(net)

    return net

    """asd"""
    # checkpoint_path = config['training']['checkpoint_path']
    # if checkpoint_path is not None:
    #     if not os.path.exists(checkpoint_path):
    #         raise FileNotFoundError(f'checkpoint: {checkpoint_path} not found')
    #
    #     dict = torch.load(checkpoint_path)
    #
    #     state_dict = dict['state_dict']
    #     # for key in list(state_dict.keys()):
    #     # if 'net.' in key:
    #     #     new_key = key.replace('net.', '')
    #     #     state_dict[new_key] = state_dict.pop(key)
    #     # if 'criterion.' in key:
    #     #     state_dict.pop(key)
    #
    #     for key, values in state_dict.items():
    #         print(key, values.size())
    #
    #     net_state_dict = net.state_dict()
    #     print('----------')
    #
    #     for key, values in net_state_dict.items():
    #         print(key, values.size())
    #
    #     tag = 'module.conv_blocks_context.'
    #
    #     encoder_dict = {
    #         '0.blocks.0.conv.weight': "input_block.conv1.conv.weight",  # torch.Size([30, 4, 3, 3, 3])
    #         '0.blocks.0.conv.bias': "input_block.conv1.conv.bias",  # torch.Size([30])
    #         '0.blocks.0.instnorm.weight': "input_block.norm1.weight",  # torch.Size([30])
    #         '0.blocks.0.instnorm.bias': "input_block.norm1.bias",  # torch.Size([30])
    #
    #         '0.blocks.1.conv.weight': "input_block.conv2.conv.weight",  # torch.Size([30, 30, 3, 3, 3])
    #         '0.blocks.1.conv.bias': "input_block.conv2.conv.bias",  # torch.Size([30])
    #         '0.blocks.1.instnorm.weight': "input_block.norm2.weight",  # torch.Size([30])
    #         '0.blocks.1.instnorm.bias': "input_block.norm2.bias",  # torch.Size([30])
    #
    #         '1.blocks.0.conv.weight': "downsamples.0.conv1.conv.weight",  # torch.Size([60, 30, 3, 3, 3])
    #         '1.blocks.0.conv.bias': "downsamples.0.conv1.conv.bias",  # torch.Size([60])
    #         '1.blocks.0.instnorm.weight': "downsamples.0.norm1.weight",  # torch.Size([60])
    #         '1.blocks.0.instnorm.bias': "downsamples.0.norm1.bias",  # torch.Size([60])
    #
    #         '1.blocks.1.conv.weight': "downsamples.0.conv2.conv.weight",  # torch.Size([60, 60, 3, 3, 3])
    #         '1.blocks.1.conv.bias': "downsamples.0.conv2.conv.bias",  # torch.Size([60])
    #         '1.blocks.1.instnorm.weight': "downsamples.0.norm2.weight",  # torch.Size([60])
    #         '1.blocks.1.instnorm.bias': "downsamples.0.norm2.bias",  # torch.Size([60])
    #
    #         '2.blocks.0.conv.weight': "downsamples.1.conv1.conv.weight",  # torch.Size([120, 60, 3, 3, 3])
    #         '2.blocks.0.conv.bias': "downsamples.1.conv1.conv.bias",  # torch.Size([120])
    #         '2.blocks.0.instnorm.weight': "downsamples.1.norm1.weight",  # torch.Size([120])
    #         '2.blocks.0.instnorm.bias': "downsamples.1.norm1.bias",  # torch.Size([120])
    #
    #         '2.blocks.1.conv.weight': "downsamples.1.conv2.conv.weight",  # torch.Size([120, 120, 3, 3, 3])
    #         '2.blocks.1.conv.bias': "downsamples.1.conv2.conv.bias",  # torch.Size([120])
    #         '2.blocks.1.instnorm.weight': "downsamples.1.norm2.weight",  # torch.Size([120])
    #         '2.blocks.1.instnorm.bias': "downsamples.1.norm2.bias",  # torch.Size([120])
    #
    #         '3.blocks.0.conv.weight': "downsamples.2.conv1.conv.weight",  # torch.Size([240, 120, 3, 3, 3])
    #         '3.blocks.0.conv.bias': "downsamples.2.conv1.conv.bias",  # torch.Size([240])
    #         '3.blocks.0.instnorm.weight': "downsamples.2.norm1.weight",  # torch.Size([240])
    #         '3.blocks.0.instnorm.bias': "downsamples.2.norm1.bias",  # torch.Size([240])
    #
    #         '3.blocks.1.conv.weight': "downsamples.2.conv2.conv.weight",  # torch.Size([240, 240, 3, 3, 3])
    #         '3.blocks.1.conv.bias': "downsamples.2.conv2.conv.bias",  # torch.Size([240])
    #         '3.blocks.1.instnorm.weight': "downsamples.2.norm2.weight",  # torch.Size([240])
    #         '3.blocks.1.instnorm.bias': "downsamples.2.norm2.bias",  # torch.Size([240])
    #
    #         '4.blocks.0.conv.weight': "downsamples.3.conv1.conv.weight",  # torch.Size([320, 240, 3, 3, 3])
    #         '4.blocks.0.conv.bias': "downsamples.3.conv1.conv.bias",  # torch.Size([320])
    #         '4.blocks.0.instnorm.weight': "downsamples.3.norm1.weight",  # torch.Size([320])
    #         '4.blocks.0.instnorm.bias': "downsamples.3.norm1.bias",  # torch.Size([320])
    #
    #         '4.blocks.1.conv.weight': "downsamples.3.conv2.conv.weight",  # torch.Size([320, 320, 3, 3, 3])
    #         '4.blocks.1.conv.bias': "downsamples.3.conv2.conv.bias",  # torch.Size([320])
    #         '4.blocks.1.instnorm.weight': "downsamples.3.norm2.weight",  # torch.Size([320])
    #         '4.blocks.1.instnorm.bias': "downsamples.3.norm2.bias",  # torch.Size([320])
    #
    #         '5.0.blocks.0.conv.weight': "bottleneck.conv1.conv.weight",  # torch.Size([320, 320, 3, 3, 3])
    #         '5.0.blocks.0.conv.bias': "bottleneck.conv1.conv.bias",  # torch.Size([320])
    #         '5.0.blocks.0.instnorm.weight': "bottleneck.norm1.weight",  # torch.Size([320])
    #         '5.0.blocks.0.instnorm.bias': "bottleneck.norm1.bias",  # torch.Size([320])
    #
    #         '5.1.blocks.0.conv.weight': "bottleneck.conv2.conv.weight",  # torch.Size([320, 320, 3, 3, 3])
    #         '5.1.blocks.0.conv.bias': "bottleneck.conv2.conv.bias",  # torch.Size([320])
    #         '5.1.blocks.0.instnorm.weight': "bottleneck.norm2.weight",  # torch.Size([320])
    #         '5.1.blocks.0.instnorm.bias': "bottleneck.norm2.bias",  # torch.Size([320])
    #
    #         'module.tu.0.weight': "upsamples.0.transp_conv.conv.weight",  # torch.Size([320, 320, 2, 2, 2])
    #         'module.tu.1.weight': "upsamples.1.transp_conv.conv.weight",  # torch.Size([320, 240, 2, 2, 2])
    #         'module.tu.2.weight': "upsamples.2.transp_conv.conv.weight",  # torch.Size([240, 120, 2, 2, 2])
    #         'module.tu.3.weight': "upsamples.3.transp_conv.conv.weight",  # torch.Size([120, 60, 2, 2, 2])
    #         'module.tu.4.weight': "upsamples.4.transp_conv.conv.weight",  # torch.Size([60, 30, 2, 2, 2])
    #     }
    #
    #     encoder_mapping = {}
    #     for key, values in encoder_dict.items():
    #         encoder_mapping[tag + key] = values
    #
    #     tag = 'module.conv_blocks_localization.'
    #     decoder_dict = {
    #         '0.0.blocks.0.conv.weight': "upsamples.0.conv_block.conv1.conv.weight",  # torch.Size([320, 640, 3, 3, 3])
    #         '0.0.blocks.0.conv.bias': "upsamples.0.conv_block.conv1.conv.bias",  # torch.Size([320])
    #         '0.0.blocks.0.instnorm.weight': "upsamples.0.conv_block.norm1.weight",  # torch.Size([320])
    #         '0.0.blocks.0.instnorm.bias': "upsamples.0.conv_block.norm1.bias",  # torch.Size([320])
    #
    #         '0.1.blocks.0.conv.weight': "upsamples.0.conv_block.conv2.conv.weight",  # torch.Size([320, 320, 3, 3, 3])
    #         '0.1.blocks.0.conv.bias': "upsamples.0.conv_block.conv2.conv.bias",  # torch.Size([320])
    #         '0.1.blocks.0.instnorm.weight': "upsamples.0.conv_block.norm2.weight",  # torch.Size([320])
    #         '0.1.blocks.0.instnorm.bias': "upsamples.0.conv_block.norm2.bias",  # torch.Size([320])
    #
    #         '1.0.blocks.0.conv.weight': "upsamples.1.conv_block.conv1.conv.weight",  # torch.Size([240, 480, 3, 3, 3])
    #         '1.0.blocks.0.conv.bias': "upsamples.1.conv_block.conv1.conv.bias",  # torch.Size([240])
    #         '1.0.blocks.0.instnorm.weight': "upsamples.1.conv_block.norm1.weight",  # torch.Size([240])
    #         '1.0.blocks.0.instnorm.bias': "upsamples.1.conv_block.norm1.bias",  # torch.Size([240])
    #
    #         '1.1.blocks.0.conv.weight': "upsamples.1.conv_block.conv2.conv.weight",  # torch.Size([240, 240, 3, 3, 3])
    #         '1.1.blocks.0.conv.bias': "upsamples.1.conv_block.conv2.conv.bias",  # torch.Size([240])
    #         '1.1.blocks.0.instnorm.weight': "upsamples.1.conv_block.norm2.weight",  # torch.Size([240])
    #         '1.1.blocks.0.instnorm.bias': "upsamples.1.conv_block.norm2.bias",  # torch.Size([240])
    #
    #         '2.0.blocks.0.conv.weight': "upsamples.2.conv_block.conv1.conv.weight",  # torch.Size([120, 240, 3, 3, 3])
    #         '2.0.blocks.0.conv.bias': "upsamples.2.conv_block.conv1.conv.bias",  # torch.Size([120])
    #         '2.0.blocks.0.instnorm.weight': "upsamples.2.conv_block.norm1.weight",  # torch.Size([120])
    #         '2.0.blocks.0.instnorm.bias': "upsamples.2.conv_block.norm1.bias",  # torch.Size([120])
    #
    #         '2.1.blocks.0.conv.weight': "upsamples.2.conv_block.conv2.conv.weight",  # torch.Size([120, 120, 3, 3, 3])
    #         '2.1.blocks.0.conv.bias': "upsamples.2.conv_block.conv2.conv.bias",  # torch.Size([120])
    #         '2.1.blocks.0.instnorm.weight': "upsamples.2.conv_block.norm2.weight",  # torch.Size([120])
    #         '2.1.blocks.0.instnorm.bias': "upsamples.2.conv_block.norm2.bias",  # torch.Size([120])
    #
    #         '3.0.blocks.0.conv.weight': "upsamples.3.conv_block.conv1.conv.weight",  # torch.Size([60, 120, 3, 3, 3])
    #         '3.0.blocks.0.conv.bias': "upsamples.3.conv_block.conv1.conv.bias",  # torch.Size([60])
    #         '3.0.blocks.0.instnorm.weight': "upsamples.3.conv_block.norm1.weight",  # torch.Size([60])
    #         '3.0.blocks.0.instnorm.bias': "upsamples.3.conv_block.norm1.bias",  # torch.Size([60])
    #
    #         '3.1.blocks.0.conv.weight': "upsamples.3.conv_block.conv2.conv.weight",  # torch.Size([60, 60, 3, 3, 3])
    #         '3.1.blocks.0.conv.bias': "upsamples.3.conv_block.conv2.conv.bias",  # torch.Size([60])
    #         '3.1.blocks.0.instnorm.weight': "upsamples.3.conv_block.norm2.weight",  # torch.Size([60])
    #         '3.1.blocks.0.instnorm.bias': "upsamples.3.conv_block.norm2.bias",  # torch.Size([60])
    #
    #         '4.0.blocks.0.conv.weight': "upsamples.4.conv_block.conv1.conv.weight",  # torch.Size([30, 60, 3, 3, 3])
    #         '4.0.blocks.0.conv.bias': "upsamples.4.conv_block.conv1.conv.bias",  # torch.Size([30])
    #         '4.0.blocks.0.instnorm.weight': "upsamples.4.conv_block.norm1.weight",  # torch.Size([30])
    #         '4.0.blocks.0.instnorm.bias': "upsamples.4.conv_block.norm1.bias",  # torch.Size([30])
    #
    #         '4.1.blocks.0.conv.weight': "upsamples.4.conv_block.conv2.conv.weight",  # torch.Size([30, 30, 3, 3, 3])
    #         '4.1.blocks.0.conv.bias': "upsamples.4.conv_block.conv2.conv.bias",  # torch.Size([30])
    #         '4.1.blocks.0.instnorm.weight': "upsamples.4.conv_block.norm2.weight",  # torch.Size([30])
    #         '4.1.blocks.0.instnorm.bias': "upsamples.4.conv_block.norm2.bias",  # torch.Size([30])
    #
    #         'module.tu.0.weight': "upsamples.0.transp_conv.conv.weight",  # torch.Size([320, 320, 2, 2, 2])
    #         'module.tu.1.weight': "upsamples.1.transp_conv.conv.weight",  # torch.Size([320, 240, 2, 2, 2])
    #         'module.tu.2.weight': "upsamples.2.transp_conv.conv.weight",  # torch.Size([240, 120, 2, 2, 2])
    #         'module.tu.3.weight': "upsamples.3.transp_conv.conv.weight",  # torch.Size([120, 60, 2, 2, 2])
    #         'module.tu.4.weight': "upsamples.4.transp_conv.conv.weight",  # torch.Size([60, 30, 2, 2, 2])
    #     }
    #
    #     decoder_mapping = {}
    #     for key, values in decoder_dict.items():
    #         decoder_mapping[tag + key] = values
    #
    # """"""

    # weight_mapping_encoder = {
    #     "module.conv_blocks_context.0.blocks.0.conv.weight": "input_block.conv1.conv.weight",
    #     "module.conv_blocks_context.0.blocks.0.conv.bias": "input_block.conv1.conv.bias",
    #     "module.conv_blocks_context.0.blocks.0.instnorm.weight": "input_block.norm1.weight",
    #     "module.conv_blocks_context.0.blocks.0.instnorm.bias": "input_block.norm1.bias",
    #     "module.conv_blocks_context.0.blocks.1.conv.weight": "input_block.conv2.conv.weight",
    #     "module.conv_blocks_context.0.blocks.1.conv.bias": "input_block.conv2.conv.bias",
    #     "module.conv_blocks_context.0.blocks.1.instnorm.weight": "input_block.norm2.weight",
    #     "module.conv_blocks_context.0.blocks.1.instnorm.bias": "input_block.norm2.bias",
    #
    #     "module.conv_blocks_context.1.blocks.0.conv.weight": "downsamples.0.conv1.conv.weight",
    #     "module.conv_blocks_context.1.blocks.0.conv.bias": "downsamples.0.conv1.conv.bias",
    #     "module.conv_blocks_context.1.blocks.0.instnorm.weight": "downsamples.0.norm1.weight",
    #     "module.conv_blocks_context.1.blocks.0.instnorm.bias": "downsamples.0.norm1.bias",
    #     "module.conv_blocks_context.1.blocks.1.conv.weight": "downsamples.0.conv2.conv.weight",
    #     "module.conv_blocks_context.1.blocks.1.conv.bias": "downsamples.0.conv2.conv.bias",  # torch.Size([60]),
    #     "module.conv_blocks_context.1.blocks.1.instnorm.weight": "downsamples.0.norm2.weight",  # torch.Size([60]),
    #     "module.conv_blocks_context.1.blocks.1.instnorm.bias": "downsamples.0.norm2.bias",  # torch.Size([60]),
    #
    #     "module.conv_blocks_context.2.blocks.0.conv.weight": "downsamples.1.conv1.conv.weight",
    #     "module.conv_blocks_context.2.blocks.0.conv.bias": "downsamples.1.conv1.conv.bias",  # torch.Size([120]),
    #     "module.conv_blocks_context.2.blocks.0.instnorm.weight": "downsamples.1.norm1.weight",  # torch.Size([120]),
    #     "module.conv_blocks_context.2.blocks.0.instnorm.bias": "downsamples.1.norm1.bias",  # torch.Size([120]),
    #     "module.conv_blocks_context.2.blocks.1.conv.weight": "downsamples.1.conv2.conv.weight",
    #     "module.conv_blocks_context.2.blocks.1.conv.bias": "downsamples.1.norm1.bias",  # torch.Size([120]),
    #     "module.conv_blocks_context.2.blocks.1.instnorm.weight": "downsamples.1.norm2.weight",  # torch.Size([120]),
    #     "module.conv_blocks_context.2.blocks.1.instnorm.bias": "downsamples.1.norm2.bias",  # torch.Size([120]),
    #
    #     "module.conv_blocks_context.3.blocks.0.conv.weight": "downsamples.2.conv1.conv.weight",
    #     "module.conv_blocks_context.3.blocks.0.conv.bias": "downsamples.2.conv1.conv.bias",  # torch.Size([120]),
    #     "module.conv_blocks_context.3.blocks.0.instnorm.weight": "downsamples.2.norm1.weight",  # torch.Size([120]),
    #     "module.conv_blocks_context.3.blocks.0.instnorm.bias": "downsamples.2.norm1.bias",  # torch.Size([120]),
    #     "module.conv_blocks_context.3.blocks.1.conv.weight": "downsamples.2.conv2.conv.weight",
    #     "module.conv_blocks_context.3.blocks.1.conv.bias": "downsamples.2.norm1.bias",  # torch.Size([120]),
    #     "module.conv_blocks_context.3.blocks.1.instnorm.weight": "downsamples.2.norm2.weight",  # torch.Size([120]),
    #     "module.conv_blocks_context.3.blocks.1.instnorm.bias": "downsamples.2.norm2.bias",  # torch.Size([120]),
    #
    #     "module.conv_blocks_context.4.blocks.0.conv.weight": "downsamples.3.conv1.conv.weight",
    #     "module.conv_blocks_context.4.blocks.0.conv.bias": "downsamples.3.conv1.conv.bias",  # torch.Size([120]),
    #     "module.conv_blocks_context.4.blocks.0.instnorm.weight": "downsamples.3.norm1.weight",  # torch.Size([120]),
    #     "module.conv_blocks_context.4.blocks.0.instnorm.bias": "downsamples.3.norm1.bias",  # torch.Size([120]),
    #     "module.conv_blocks_context.4.blocks.1.conv.weight": "downsamples.3.conv2.conv.weight",
    #     "module.conv_blocks_context.4.blocks.1.conv.bias": "downsamples.3.norm1.bias",  # torch.Size([120]),
    #     "module.conv_blocks_context.4.blocks.1.instnorm.weight": "downsamples.3.norm2.weight",  # torch.Size([120]),
    #     "module.conv_blocks_context.4.blocks.1.instnorm.bias": "downsamples.3.norm2.bias",  # torch.Size([120]),
    #
    #     "module.conv_blocks_context.5.blocks.0.conv.weight": "bottleneck.conv1.conv.weight",
    #     "module.conv_blocks_context.5.blocks.0.conv.bias": "bottleneck.conv1.conv.bias",  # torch.Size([120]),
    #     "module.conv_blocks_context.5.blocks.0.instnorm.weight": "bottleneck.norm1.weight",  # torch.Size([120]),
    #     "module.conv_blocks_context.5.blocks.0.instnorm.bias": "bottleneck.norm1.bias",  # torch.Size([120]),
    #     "module.conv_blocks_context.5.blocks.1.conv.weight": "bottleneck.conv2.conv.weight",
    #     "module.conv_blocks_context.5.blocks.1.conv.bias": "bottleneck.norm1.bias",  # torch.Size([120]),
    #     "module.conv_blocks_context.5.blocks.1.instnorm.weight": "bottleneck.norm2.weight",  # torch.Size([120]),
    #     "module.conv_blocks_context.5.blocks.1.instnorm.bias": "bottleneck.norm2.bias",  # torch.Size([120]),
    # }

    # weight_mapping_decoder = {
    #     "module.conv_blocks_localization.0.0.blocks.0.conv.weight": "upsamples.0.conv_block.conv1.conv.weight",
    #     # "module.conv_blocks_localization.0.0.blocks.0.conv.bias": "upsamples.0.conv_block.norm1.bias",  # torch.Size([320]),
    #     "module.conv_blocks_localization.0.0.blocks.0.instnorm.weight": "upsamples.0.conv_block.norm1.weight",
    #     "module.conv_blocks_localization.0.0.blocks.0.instnorm.bias": "upsamples.0.conv_block.norm1.bias",
    #
    #     "module.conv_blocks_localization.0.1.blocks.0.conv.weight": "upsamples.0.conv_block.conv2.conv.weight",
    #     # "module.conv_blocks_localization.0.1.blocks.0.conv.bias": "upsamples.0.transp_conv.conv.bias", # torch.Size([320]),
    #     "module.conv_blocks_localization.0.1.blocks.0.instnorm.weight": "upsamples.0.conv_block.norm2.weight",
    #     "module.conv_blocks_localization.0.1.blocks.0.instnorm.bias": "upsamples.0.conv_block.norm2.bias",
    #
    #     "module.conv_blocks_localization.1.0.blocks.0.conv.weight": "upsamples.1.conv_block.conv1.conv.weight",
    #     # "module.conv_blocks_localization.1.0.blocks.0.conv.bias": "",  # torch.Size([240]),
    #     "module.conv_blocks_localization.1.0.blocks.0.instnorm.weight": "upsamples.1.conv_block.norm1.weight",
    #     "module.conv_blocks_localization.1.0.blocks.0.instnorm.bias": "upsamples.1.conv_block.norm1.bias",
    #
    #     "module.conv_blocks_localization.1.1.blocks.0.conv.weight": "upsamples.1.conv_block.conv2.conv.weight",
    #     # "module.conv_blocks_localization.1.1.blocks.0.conv.bias": "",  # torch.Size([240]),
    #     "module.conv_blocks_localization.1.1.blocks.0.instnorm.weight": "upsamples.1.conv_block.norm2.weight",
    #     "module.conv_blocks_localization.1.1.blocks.0.instnorm.bias": "upsamples.1.conv_block.norm2.bias",
    #
    #     "module.conv_blocks_localization.2.0.blocks.0.conv.weight": "upsamples.2.conv_block.conv1.conv.weight",
    #     # "module.conv_blocks_localization.2.0.blocks.0.conv.bias": "",  # torch.Size([120]),
    #     "module.conv_blocks_localization.2.0.blocks.0.instnorm.weight": "upsamples.2.conv_block.norm1.weight",
    #     "module.conv_blocks_localization.2.0.blocks.0.instnorm.bias": "upsamples.2.conv_block.norm1.bias",
    #
    #     "module.conv_blocks_localization.2.1.blocks.0.conv.weight": "upsamples.2.conv_block.conv2.conv.weight",
    #     # "module.conv_blocks_localization.2.1.blocks.0.conv.bias": "",  # torch.Size([120]),
    #     "module.conv_blocks_localization.2.1.blocks.0.instnorm.weight": "upsamples.2.conv_block.norm2.weight",
    #     "module.conv_blocks_localization.2.1.blocks.0.instnorm.bias": "upsamples.2.conv_block.norm2.bias",
    #
    #     "module.conv_blocks_localization.3.0.blocks.0.conv.weight": "upsamples.3.conv_block.conv1.conv.weight",
    #     # "module.conv_blocks_localization.3.0.blocks.0.conv.bias": "",  # torch.Size([60]),
    #     "module.conv_blocks_localization.3.0.blocks.0.instnorm.weight": "upsamples.3.conv_block.norm1.weight",
    #     "module.conv_blocks_localization.3.0.blocks.0.instnorm.bias": "upsamples.3.conv_block.norm1.bias",
    #
    #     "module.conv_blocks_localization.3.1.blocks.0.conv.weight": "upsamples.3.conv_block.conv2.conv.weight",
    #     # "module.conv_blocks_localization.3.1.blocks.0.conv.bias": "",  # torch.Size([60]),
    #     "module.conv_blocks_localization.3.1.blocks.0.instnorm.weight": "upsamples.3.conv_block.norm2.weight",
    #     "module.conv_blocks_localization.3.1.blocks.0.instnorm.bias": "upsamples.3.conv_block.norm2.bias",
    #
    #     "module.conv_blocks_localization.4.0.blocks.0.conv.weight": "upsamples.4.conv_block.conv1.conv.weight",
    #     # "module.conv_blocks_localization.4.0.blocks.0.conv.bias": "",
    #     "module.conv_blocks_localization.4.0.blocks.0.instnorm.weight": "upsamples.4.conv_block.norm1.weight",
    #     "module.conv_blocks_localization.4.0.blocks.0.instnorm.bias": "upsamples.4.conv_block.norm1.bias",
    #
    #     "module.conv_blocks_localization.4.1.blocks.0.conv.weight": "upsamples.4.conv_block.conv2.conv.weight",
    #     # "module.conv_blocks_localization.4.1.blocks.0.conv.bias": "",
    #     "module.conv_blocks_localization.4.1.blocks.0.instnorm.weight": "upsamples.4.conv_block.norm2.weight",
    #     "module.conv_blocks_localization.4.1.blocks.0.instnorm.bias": "upsamples.4.conv_block.norm2.bias",
    # }

    """gut"""

    # # Overwrite encoder
    # for old_key, new_key in encoder_mapping.items():
    #     if old_key in state_dict:
    #         net_state_dict[new_key] = state_dict[old_key]
    #
    # # Overwrite decoder
    # # for old_key, new_key in decoder_mapping.items():
    # #     if old_key in state_dict:
    # #         net_state_dict[new_key] = state_dict[old_key]
    # #
    # net.load_state_dict(net_state_dict)
    #
    # freeze_encoder = [
    #     "input_block.conv1.conv.weight", "input_block.conv1.conv.bias",
    #     "input_block.conv2.conv.weight", "input_block.conv2.conv.bias",
    #     "input_block.norm1.weight", "input_block.norm1.bias",
    #     "input_block.norm2.weight", "input_block.norm2.bias",
    #
    #     "downsamples.0.conv1.conv.weight", "downsamples.0.conv1.conv.bias",
    #     "downsamples.0.conv2.conv.weight", "downsamples.0.conv2.conv.bias",
    #     "downsamples.0.norm1.weight", "downsamples.0.norm1.bias",
    #     "downsamples.0.norm2.weight", "downsamples.0.norm2.bias",
    #
    #     "downsamples.1.conv1.conv.weight", "downsamples.1.conv1.conv.bias",
    #     "downsamples.1.conv2.conv.weight", "downsamples.1.conv2.conv.bias",
    #     "downsamples.1.norm1.weight", "downsamples.1.norm1.bias",
    #     "downsamples.1.norm2.weight", "downsamples.1.norm2.bias",
    #
    #     "downsamples.2.conv1.conv.weight", "downsamples.2.conv1.conv.bias",
    #     "downsamples.2.conv2.conv.weight", "downsamples.2.conv2.conv.bias",
    #     "downsamples.2.norm1.weight", "downsamples.2.norm1.bias",
    #     "downsamples.2.norm2.weight", "downsamples.2.norm2.bias",
    #
    #     "downsamples.3.conv1.conv.weight", "downsamples.3.conv1.conv.bias",
    #     "downsamples.3.conv2.conv.weight", "downsamples.3.conv2.conv.bias",
    #     "downsamples.3.norm1.weight", "downsamples.3.norm1.bias",
    #     "downsamples.3.norm2.weight", "downsamples.3.norm2.bias",
    #
    #     "bottleneck.conv1.conv.weight", "bottleneck.conv1.conv.bias",
    #     "bottleneck.conv2.conv.weight", "bottleneck.conv2.conv.bias",
    #     "bottleneck.norm1.weight", "bottleneck.norm1.bias",
    #     "bottleneck.norm2.weight", "bottleneck.norm2.bias",
    #
    #     'upsamples.0.transp_conv.conv.weight', 'upsamples.1.transp_conv.conv.weight',
    #     'upsamples.2.transp_conv.conv.weight', 'upsamples.3.transp_conv.conv.weight',
    #     'upsamples.4.transp_conv.conv.weight'
    # ]
    #
    # for name, param in net.named_parameters():
    #     if name in freeze_encoder:
    #         param.requires_grad = False

    """"""
    # freeze_decoder = [
    #     'upsamples.0.conv_block.conv1.conv.weight', 'upsamples.0.conv_block.norm1.weight',
    #     'upsamples.0.conv_block.norm1.bias', 'upsamples.0.conv_block.conv2.conv.weight',
    #     'upsamples.0.conv_block.norm2.weight', 'upsamples.0.conv_block.norm2.bias',
    #
    #     'upsamples.1.conv_block.conv1.conv.weight', 'upsamples.1.conv_block.norm1.weight',
    #     'upsamples.1.conv_block.norm1.bias', 'upsamples.1.conv_block.conv2.conv.weight',
    #     'upsamples.1.conv_block.norm2.weight', 'upsamples.1.conv_block.norm2.bias',
    #
    #     'upsamples.2.conv_block.conv1.conv.weight', 'upsamples.2.conv_block.norm1.weight',
    #     'upsamples.2.conv_block.norm1.bias', 'upsamples.2.conv_block.conv2.conv.weight',
    #     'upsamples.2.conv_block.norm2.weight', 'upsamples.2.conv_block.norm2.bias',
    #
    #     'upsamples.3.conv_block.conv1.conv.weight', 'upsamples.3.conv_block.norm1.weight',
    #     'upsamples.3.conv_block.norm1.bias', 'upsamples.3.conv_block.conv2.conv.weight',
    #     'upsamples.3.conv_block.norm2.weight', 'upsamples.3.conv_block.norm2.bias',
    #
    #     'upsamples.4.conv_block.conv1.conv.weight', 'upsamples.4.conv_block.norm1.weight',
    #     'upsamples.4.conv_block.norm1.bias', 'upsamples.4.conv_block.conv2.conv.weight',
    #     'upsamples.4.conv_block.norm2.weight', 'upsamples.4.conv_block.norm2.bias',
    #
    #     'upsamples.0.transp_conv.conv.weight', 'upsamples.1.transp_conv.conv.weight',
    #     'upsamples.2.transp_conv.conv.weight', 'upsamples.3.transp_conv.conv.weight',
    #     'upsamples.4.transp_conv.conv.weight'
    #
    #
    # ]

    # for name, param in net.named_parameters():
    #     if name in freeze_decoder:
    #         param.requires_grad = False
