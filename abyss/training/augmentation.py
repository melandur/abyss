import monai.transforms as tf
import torchio as tio

# Spatialf
#    tr_transforms.append(SpatialTransform_2(
#        patch_size_spatial,
#        patch_center_dist_from_border=None,
#        do_elastic_deform=params.get("do_elastic"),
#        deformation_scale=params.get("eldef_deformation_scale"),
#        - do_rotation=params.get("do_rotation"),
#        - angle_x=params.get("rotation_x"),
#        - angle_y=params.get("rotation_y"),
#        - angle_z=params.get("rotation_z"),
#        do_scale=params.get("do_scaling"),
#        scale=params.get("scale_range"),
#        border_mode_data=params.get("border_mode_data"),
#        border_cval_data=0,
#        order_data=order_data,
#        border_mode_seg="constant",
#        border_cval_seg=border_val_seg,
#        order_seg=order_seg,
#        random_crop=params.get("random_crop"),
#        p_el_per_sample=params.get("p_eldef"),
#        p_scale_per_sample=params.get("p_scale"),
#        p_rot_per_sample=params.get("p_rot"),
#        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis"),
#        p_independent_scale_per_axis=params.get("p_independent_scale_per_axis")
#     ))
#


class Augmentation:
    """Composes transformation based on the config file order"""

    def __init__(self):
        spatial_transforms = tf.Compose(
            [
                tf.RandRotate(range_x=(-1, 1), range_y=(-1, 1), range_z=(-1, 1), prob=0.8),
                # tf.OneOf([tio.RandomFlip(axes=0, flip_probability=1.0),
                #           tio.RandomFlip(axes=1, flip_probability=1.0),
                #           tio.RandomFlip(axes=2, flip_probability=1.0)]),
                # tio.RandomElasticDeformation(max_displacement=7, locked_borders=2,
                #                              image_interpolation='bspline', label_interpolation='nearest'),
                # tio.RandomElasticDeformation(max_displacement=2, locked_borders=2,
                #                              image_interpolation='bspline', label_interpolation='nearest'),
            ]
        )

        # spatial_transforms = [
        #     tf.OneOf(
        #     [
        #
        #     ]
        #
        # )]
        intensity_transforms = tf.OneOf(
            [
                tf.RandAdjustContrast(prob=1.0, gamma=(1.0, 2.0)),
                tf.RandBiasField(prob=1.0),
                tf.RandGaussianNoise(prob=1.0),
            ]
        )

        artefact_transforms = tf.OneOf(
            [
                # tf.RandGibbsNoise(prob=1.0, alpha=(0.1, 1.0), as_tensor_output=True),
                # tf.RandKSpaceSpikeNoise(prob=1.0, intensity_range=(12, 12), channel_wise=False,
                # as_tensor_output=True),
                tio.RandomMotion(translation=0.1, degrees=2)
            ]
        )

        self.transform = tf.Compose(
            [
                # orientation_transforms,,
                spatial_transforms,
                # intensity_transforms,
                # artefact_transforms,
                # tf.RandCoarseDropout(holes=100, spatial_size=10, fill_value=0, prob=0.5),
                # tf.NormalizeIntensity(),
            ]
        )

    def __call__(self):
        return self.transform

        # if tr_name == 'RandGaussianNoise':
        #     transforms.append(tf.RandGaussianNoise(**tr_params, dtype=np.float32))
        # elif tr_name == 'RandGaussianSmooth':
        #     transforms.append(tf.RandGaussianSmooth(**tr_params))
        # elif tr_name == 'RandScaleIntensity':
        #     transforms.append(tf.RandScaleIntensity(**tr_params, dtype=np.float32))
        # elif tr_name == 'RandFlip':
        #     transforms.append(tf.RandFlip(**tr_params))
        # elif tr_name == 'RandAdjustContrast':
        #     transforms.append(tf.RandAdjustContrast(**tr_params))
        # elif tr_name == 'RandRotate':
        #     transforms.append(tf.RandRotate(**tr_params, dtype=np.float64))
        # elif tr_name == 'RandScaleCrop':
        #     transforms.append(tf.RandScaleCrop(**tr_params))
        # elif tr_name == 'RandHistogramShift':
        #     transforms.append(tf.RandHistogramShift(**tr_params))
        # elif tr_name == 'RandSpatialCrop':
        #     transforms.append(tf.RandSpatialCrop(**tr_params))
        # elif tr_name == 'RandBiasField':
        #     transforms.append(tf.RandBiasField(**tr_params, dtype=np.float32))
        # elif tr_name == 'RandRotate90':
        #     transforms.append(tf.RandRotate90(**tr_params))
        # elif tr_name == 'RandGibbsNoise':
        #     transforms.append(tf.RandGibbsNoise(**tr_params))
        # elif tr_name == 'RandKSpaceSpikeNoise':
        #     transforms.append(tf.RandKSpaceSpikeNoise(**tr_params))
        # elif tr_name == 'RandCoarseDropout':
        #     transforms.append(tf.RandCoarseDropout(**tr_params))
        # else:
        #     raise NotImplementedError(f'{tr_name} is missing, add transformation -> training -> augmentation.py')


if __name__ == '__main__':
    aug = Augmentation()
