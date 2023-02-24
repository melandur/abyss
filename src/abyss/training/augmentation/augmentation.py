import monai.transforms as tf
import torchio as tio

# from abyss.training.augmentation.custom_augmentations import (
#     RandomChannelDropout,
#     RandomChannelShuffle,
# )

spatial_transforms = tf.Compose(
    [
        # tf.RandSpatialCropd(
        #     keys=['data', 'label'],
        #     roi_size=(80, 80, 80),
        #     max_roi_size=None,
        #     random_center=True,
        #     random_size=False,
        # ),
        tf.RandRotated(
            keys=['data', 'label'],
            range_x=(-1, 1),
            range_y=(-1, 1),
            range_z=(-1, 1),
            prob=0.8,
            mode='nearest',
        ),
        tf.OneOf(
            [
                tio.RandomFlip(
                    include=['data', 'label'],
                    axes=0,
                    flip_probability=1.0,
                ),
                tio.RandomFlip(
                    include=['data', 'label'],
                    axes=1,
                    flip_probability=1.0,
                ),
                tio.RandomFlip(
                    include=['data', 'label'],
                    axes=2,
                    flip_probability=1.0,
                ),
            ]
        ),
        # tf.Rand3DElasticd(
        #     keys=['data', 'label'],
        #     sigma_range=(5, 7),
        #     shear_range=0.3,
        #     translate_range=5,
        #     magnitude_range=(50, 110),
        #     mode='nearest',
        #     padding_mode='zeros',
        #     as_tensor_output=True,
        #     prob=1.0,
        # ),
        # tf.Rand2DElasticd(
        #     keys=['data', 'label'],
        #     sigma_range=(5, 7),
        #     shear_range=0.3,
        #     translate_range=5,
        #     magnitude_range=(50, 110),
        #     mode='nearest',
        #     padding_mode='zeros',
        #     as_tensor_output=True,
        #     prob=1.0,
        # ),
    ]
)

intensity_transforms = tf.OneOf(
    [
        tf.RandAdjustContrastd(
            keys=['data'],
            gamma=(0.5, 2.0),
            prob=1.0,
        ),
        tf.RandBiasFieldd(
            keys=['data'],
            degree=3,
            coeff_range=(0.0, 0.1),
            prob=1.0,
        ),
    ]
)

artefact_transforms = tf.OneOf(
    [
        tf.RandGaussianNoised(
            keys=['data'],
            prob=1.0,
        ),
        tio.RandomBlur(
            include=['data'],
            std=(0, 2),
        ),
        tf.RandGibbsNoised(
            keys=['data'],
            prob=1.0,
            alpha=(0.3, 0.7),
            as_tensor_output=True,
        ),
        tf.RandKSpaceSpikeNoised(
            keys=['data'],
            prob=1.0,
            intensity_range=(1, 11),
            channel_wise=False,
            as_tensor_output=True,
        ),
        tio.RandomMotion(
            include=['data'],
            translation=0.05,
            degrees=2,
            image_interpolation='Bspline',
            num_transforms=1,
        ),
        tio.RandomGhosting(
            include='data',
            num_ghosts=(4, 10),
            axes=(0, 1, 2),
            intensity=(0.5, 1),
        ),
    ]
)

transforms = tf.Compose(
    [
        # spatial_transforms,
        intensity_transforms,
        artefact_transforms,
        # RandomChannelDropout(include=['data'], num_channels=1, fill_value=0.0, prob=0.8),
        # RandomChannelShuffle(include=['data'], prob=1.0),
        tf.ScaleIntensityd(keys=['data'], minv=0.0, maxv=1.0),
        # tio.OneHot(keys=['label'], num_classes=1)
    ]
)
