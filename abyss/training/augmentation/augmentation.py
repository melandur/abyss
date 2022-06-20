import monai.transforms as tf
import torchio as tio

from abyss.training.augmentation.custom_augmentations import (
    RandomChannelShuffle,
    RandomChannelSkip,
)

# Spatialf
#    tr_transforms.append(SpatialTransform_2(
#        patch_center_dist_from_border=None,
#        deformation_scale=params.get("eldef_deformation_scale"),
#        - do_rotation=params.get("do_rotation"),
#        - angle_x=params.get("rotation_x"),
#        - angle_y=params.get("rotation_y"),
#        - angle_z=params.get("rotation_z"),
#        do_scale=params.get("do_scaling"),
#        scale=params.get("scale_range"),
#        border_mode_data=params.get("border_mode_data"),
#        border_cval_data=0,
#        border_mode_seg="constant",
#        border_cval_seg=border_val_seg,
#        p_el_per_sample=params.get("p_eldef"),
#        p_scale_per_sample=params.get("p_scale")
#        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis"),
#        p_independent_scale_per_axis=params.get("p_independent_scale_per_axis")
#     ))
#


class Augmentation:
    """Composes transformation based on the config file order"""

    def __init__(self):
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
                tf.Rand3DElasticd(
                    keys=['data', 'label'],
                    sigma_range=(5, 7),
                    shear_range=0.5,
                    translate_range=5,
                    magnitude_range=(50, 110),
                    mode='nearest',
                    padding_mode='zeros',
                    as_tensor_output=True,
                    prob=1.0,
                ),
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

        self.data_transforms = tf.Compose(
            [
                # RandomChannelSkip(include=['data'], num_channels=1, fill_value=0.0, prob=0.5)
                RandomChannelShuffle(include=['data'], prob=1)
                # tio.RandomAffine(include=['data', 'label'], image_interpolation='bspline',
                #                  label_interpolation='label_gaussian'),
                # tf.RandAffined(keys=['data', 'label'], translate_range=2, shear_range=1, prob=1.0)
                # tf.Rand3DElasticd(keys=['data', 'label'], sigma_range=(12, 17), magnitude_range=(50, 300),
                #                   shear_range=1, translate_range=1, scale_range=1)
                # spatial_transforms,
                # intensity_transforms,
                # artefact_transforms,
                # tf.NormalizeIntensityd(keys=['data']),
                # tf.RandCoarseDropoutd(keys=['data'], holes=100, spatial_size=10, fill_value=0.0, prob=0.5),
            ]
        )

    def __call__(self):
        return self.data_transforms


if __name__ == '__main__':
    aug = Augmentation()
