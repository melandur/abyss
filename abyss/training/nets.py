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
