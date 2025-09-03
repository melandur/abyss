import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

torch._dynamo.config.cache_size_limit = 24  # increase cache size limit


from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from abyss.training.network_definitions import DynUNet


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

    net = load_pretrained_weights(
        net,
        pretrained_weights_path='/home/melandur/code/abyss/data/training/checkpoint_final_ResEncL_MAE.pth',
        pt_input_channels=1,
        downstream_input_channels=in_channels,
        pt_input_patchsize=[160, 160, 160],
        downstream_input_patchsize=config['trainer']['patch_size'][0],
        pt_key_to_encoder='encoder.stages',
        pt_key_to_stem='encoder.stem',
        pt_keys_to_in_proj=["encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"],
        pt_key_to_lpe=None,
    )[0]

    if config['training']['compile']:
        net = torch.compile(net)

    return net


def load_pretrained_weights(
    network,
    pretrained_weights_path: str,
    pt_input_channels: int,
    downstream_input_channels: int,
    pt_input_patchsize: int,
    downstream_input_patchsize: int,
    pt_key_to_encoder: str,
    pt_key_to_stem: str,
    pt_keys_to_in_proj: tuple[str, ...],
    pt_key_to_lpe: str,
) -> tuple[nn.Module, bool]:
    """
    Load pretrained weights into the network.
    Per default we only load the encoder and the stem weights. The stem weights are adapted to the number of input channels through repeats.
    The decoder is initialized from scratch.

    :param network: The neural network to load weights into.
    :param pretrained_weights_path: Path to the pretrained weights file.
    :param pt_input_channels: Number of input channels used in the pretrained model.
    :param downstream_input_channels: Number of input channels used during adaptation (currently).
    :param pt_input_patchsize: Patch size used in the pretrained model.
    :param downstream_input_patchsize: Patch size used during adaptation (currently).
    :param pt_key_to_encoder: Key to the encoder in the pretrained model.
    :param pt_key_to_stem: Key to the stem in the pretrained model.

    :return: The network with loaded weights.
    """

    # --------------------------- Technical Description -------------------------- #
    # In this function we want to load the weights in a reliable manner.
    #   Hence we want to load the weights with `strict=False` to guarantee everything is loaded as expected.
    #   To do so, we grab the respective submodules and load the fitting weights into them.
    #   We can do this through `get_submodule` which is a nn.Module function.
    #   However we need to cut-away the prefix of the matching keys to correctly assign weights from both `state_dicts`!
    # Difficulties:
    # 1) Different stem dimensions: When pre-training had only a single input channel, we need to make the shapes fit!
    #    To do so, we utilize repeating the weights N times (N = number of input channels).
    #    Limitation currently we only support this for a single input channel used during pre-training.
    # 2) Different patch sizes: The learned positional embeddings LPe of `Transformer` (Primus) architectures are
    #    patch size dependent. To adapt the weights, we do trilinear interpolation of these weights back to shape.
    # 3) Stem and Encoder merging: Most architectures (Primus, ResidualEncoderUNet derivatives) have
    #    separate `stem` and `encoder` objects. Hence we can separate stem and encoder weight loading easily.
    #    However in the `PlainConvUNet` architecture the encoder contains the stem, so we must make sure
    #    to skip the stem weight loading in the encoder, and then separately load the (repeated) stem weights

    # The following code does this.
    key_to_encoder = network.key_to_encoder  # Key to the encoder in the current network
    key_to_stem = network.key_to_stem  # Key to the stem (beginning) in the current network

    random_init_statedict = network.state_dict()
    ckp = torch.load(pretrained_weights_path, weights_only=True)
    pre_train_statedict: dict[str, torch.Tensor] = ckp["network_weights"]  # Get pre-trained state dict

    # take info from ckpt path (allows to overwrite plan specifications)
    if 'key_to_stem' in ckp['nnssl_adaptation_plan'].keys():
        pt_key_to_stem = ckp['nnssl_adaptation_plan']['key_to_stem']
    if 'key_to_encoder' in ckp['nnssl_adaptation_plan'].keys():
        pt_key_to_encoder = ckp['nnssl_adaptation_plan']['key_to_encoder']
    if 'keys_to_in_proj' in ckp['nnssl_adaptation_plan'].keys():
        pt_keys_to_in_proj = ckp['nnssl_adaptation_plan']['keys_to_in_proj']
    if 'key_to_lpe' in ckp['nnssl_adaptation_plan'].keys():
        pt_key_to_lpe = ckp['nnssl_adaptation_plan']['key_to_lpe']

    ####allows overwrites (e.g for voco needed)
    if 'nnssl_adaptation_plan' in ckp.keys():
        if 'pretrain_patch_size' in ckp['nnssl_adaptation_plan'].keys():
            pt_input_patchsize = ckp['nnssl_adaptation_plan']['pretrain_patch_size']

    stem_in_encoder = pt_key_to_stem in pre_train_statedict

    # Currently we don't have the logic for interpolating the positional embedding yet.
    pt_weight_in_ch_mismatch = False
    need_to_adapt_lpe = False  # I.e. Learnable positional embedding
    key_to_lpe = getattr(network, "key_to_lpe", None)
    lpe_in_stem = False

    # # Check if the current module even uses a learnable positional embedding. If not ignore LPE logic.
    try:
        network.get_submodule(key_to_lpe)
    except AttributeError:
        key_to_lpe = None

    if key_to_lpe is not None:
        lpe_in_encoder = key_to_lpe.startswith(key_to_encoder)
        lpe_in_stem = key_to_lpe.startswith(key_to_stem)
    if pt_input_patchsize != downstream_input_patchsize:
        need_to_adapt_lpe = True  # LPE shape won't fit -> resize it

    def strip_dot_prefix(s) -> str:
        """Mini func to strip the dot prefix from the keys"""
        if s.startswith("."):
            return s[1:]
        return s

    # ----- Match the keys of pretrained weights to the current architecture ----- #
    if stem_in_encoder:
        encoder_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_encoder)}
        if downstream_input_channels > pt_input_channels:
            pt_weight_in_ch_mismatch = True
            k_proj = pt_keys_to_in_proj[0] + ".weight"  # Get the projection weights
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
                new_encoder_weights["pos_embed"].to(next(network.parameters()).device)
            if "cls_token" in encoder_weights.keys():
                skip_strings_in_pretrained = ["cls_token"]
                new_encoder_weights, found_cls_token = filter_state_dict(encoder_weights, skip_strings_in_pretrained)

        # ------------------------------- Load weights ------------------------------- #
        encoder_module = network.get_submodule(key_to_encoder)
        encoder_module.load_state_dict(new_encoder_weights)
    else:
        encoder_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_encoder)}
        stem_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_stem)}
        if downstream_input_channels > pt_input_channels:
            pt_weight_in_ch_mismatch = True
            k_proj = pt_keys_to_in_proj[0] + ".weight"  # Get the projection weights
            vals = (stem_weights[k_proj].repeat(1, downstream_input_channels, 1, 1, 1)) / downstream_input_channels
            for k in pt_keys_to_in_proj:
                stem_weights[k + ".weight"] = vals
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
                new_stem_weights["pos_embed"].to(next(network.parameters()).device)
            elif lpe_in_encoder:
                handle_pos_embed_resize(
                    new_encoder_weights,
                    network.get_submodule(key_to_encoder).state_dict(),
                    'interpolate_trilinear',
                    downstream_input_patchsize,
                    pt_input_patchsize,
                    new_stem_weights['proj.weight'].shape[2:],
                )
                new_encoder_weights["pos_embed"].to(next(network.parameters()).device)
            else:
                pass
        if "cls_token" in encoder_weights.keys():
            skip_strings_in_pretrained = ["cls_token"]
            new_encoder_weights, found_cls_token = filter_state_dict(encoder_weights, skip_strings_in_pretrained)

        # ------------------------------- Load weights ------------------------------- #
        encoder_module = network.get_submodule(key_to_encoder)
        encoder_module.load_state_dict(new_encoder_weights)
        stem_module = network.get_submodule(key_to_stem)
        stem_module.load_state_dict(new_stem_weights)

    if not need_to_adapt_lpe and key_to_lpe is not None:
        # Load the positional embedding weights
        lpe_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_lpe)}
        assert (
            len(lpe_weights) == 1
        ), f"Found multiple lpe weights, but expect only a single tensor. Got {list(lpe_weights.keys())}"
        network.get_parameter(key_to_lpe).data = list(lpe_weights.values())[0].to(next(network.parameters()).device)
        # ------------------------------- Load weights ------------------------------- #

    # Theoretically we don't need to return the network, but we do it anyway.
    del pre_train_statedict, encoder_weights, new_encoder_weights
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
    pretrained_pos_embed = pretrained_dict["pos_embed"]
    model_pos_embed = model_dict["pos_embed"]
    model_pos_embed_shape = model_pos_embed.shape

    # for key, value in pretrained_dict.items():
    #     print(f"{key}: {value.shape}")

    has_cls_token = "cls_token" in pretrained_dict

    if has_cls_token:
        cls_pos_embed = pretrained_pos_embed[:, :1, :]
        patch_pos_embed = pretrained_pos_embed[:, 1:, :]
    else:
        if "cls_token" in model_dict.keys():
            cls_pos_embed = model_pos_embed[:, :1, :]
        patch_pos_embed = pretrained_pos_embed

    if mode == "interpolate":
        resized_patch_pos_embed = interpolate_patch_embed_1d(
            patch_pos_embed, target_len=model_pos_embed_shape[1] - int(has_cls_token)
        )

    elif mode == "interpolate_trilinear":
        # Calculate input/output 3D shapes
        in_shape = dict(zip("xyz", [int(d / p) for d, p in zip(pretrained_input_patch_size, patch_embed_size)]))
        out_shape = dict(zip("xyz", [int(d / p) for d, p in zip(input_shape, patch_embed_size)]))
        resized_patch_pos_embed = interpolate_patch_embed_3d(patch_pos_embed, in_shape, out_shape)

    else:
        raise NotImplementedError(f"Unknown resize mode: {mode}")
    if "cls_token" in model_dict.keys():
        resized_pos_embed = torch.cat([cls_pos_embed, resized_patch_pos_embed], dim=1)
    else:
        resized_pos_embed = resized_patch_pos_embed
    pretrained_dict["pos_embed"] = resized_pos_embed
