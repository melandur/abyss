import torch

torch._dynamo.config.cache_size_limit = 24  # increase cache size limit

from abyss.training.network_definitions import UNETR, DynUNet
from abyss.training.primer import PrimusB, PrimusL, PrimusM, PrimusS


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

    # net = UNETR(
    #     in_channels=in_channels,
    #     out_channels=out_channels,
    #     img_size=config['trainer']['patch_size'],
    #     feature_size=32,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_heads=12,
    #     proj_type='conv',
    #     norm_name='instance',
    #     dropout_rate=0.0,
    #     qkv_bias=True,
    #     save_attn=False,
    # )

    # net = DynUNet(
    #     spatial_dims=3,
    #     in_channels=in_channels,
    #     out_channels=out_channels,
    #     filters=[64, 128, 256, 512, 512, 512],  # brats winner 22
    #     # filters=[64, 96, 128, 192, 256, 384, 512, 768, 1024],  # brats winner 21
    #     # filters=[30, 60, 120, 240, 320, 320],  # nnunet winner 18/19
    #     kernel_size=dimensions['kernel_size'],
    #     strides=dimensions['strides'],
    #     upsample_kernel_size=dimensions['upsample_kernel_size'],
    #     dropout=None,
    #     norm_name=('instance', {'affine': True}),
    #     act_name=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
    #     deep_supervision=True,
    #     deep_supr_num=2,
    #     res_block=True,
    #     trans_bias=True,
    # )

    net = PrimusB(
        input_channels=in_channels,
        output_channels=out_channels,
        patch_embed_size=(8, 8, 8),
        input_shape=config['trainer']['patch_size'],
        drop_path_rate=0.2,
    )

    if config['training']['compile']:
        net = torch.compile(net)

    return net
