import os

import torch
from monai.networks.nets import DynUNet


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
    dimemsions = get_kernels_strides(config)

    net = DynUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        kernel_size=dimemsions['kernel_size'],
        strides=dimemsions['strides'],
        upsample_kernel_size=dimemsions['upsample_kernel_size'],
        dropout=None,
        norm_name='instance',
        deep_supervision=True,
        deep_supr_num=1,
        res_block=False,
        trans_bias=False,
    )

    checkpoint_path = config['training']['checkpoint_path']
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'checkpoint: {checkpoint_path} not found')
        net.load_state_dict(torch.load(checkpoint_path))
    return net
