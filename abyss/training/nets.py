# pylint: disable-all

from monai.networks.nets import UNet

net = UNet(
    spatial_dims=128,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=0,
)
