# based on monai.transforms | comment/delete unused ones | order corresponds to the application sequence
# https://docs.monai.io/en/stable/transforms.html#vanilla-transforms

augmentation = {
    # 'RandGaussianNoise': {'prob': 1.0, 'mean': 100.0, 'std': 1.0},
    # 'RandGaussianSmooth': {
    #     'sigma_x': (0.25, 1.5),
    #     'sigma_y': (0.25, 1.5),
    #     'sigma_z': (0.25, 1.5),
    #     'prob': 0.1,
    #     'approx': 'erf',
    # },
    # 'RandScaleIntensity': {'factors': (1.0, 1.0), 'prob': 0.1},
    # 'RandFlip': {'prob': 0.1, 'spatial_axis': None},
    # 'RandAdjustContrast': {'prob': 0.1, 'gamma': (0.5, 4.5)},
    # 'RandRotate': {
    #     'range_x': 0.0,
    #     'range_y': 0.0,
    #     'range_z': 0.0,
    #     'prob': 0.1,
    #     'keep_size': True,
    #     'mode': 'bilinear',
    #     'padding_mode': 'border',
    #     'align_corners': False,
    # },
    # 'RandScaleCrop': {
    #     'roi_scale': None,
    #     'max_roi_scale': (50, 50, 50),
    #     'random_center': True,
    #     'random_size': True,
    # },
    # 'RandHistogramShift': {'num_control_points': 10, 'prob': 0.1},
    # 'RandSpatialCrop': {
    #     'roi_size': None,
    #     'max_roi_size': [100, 50, 50],
    #     'random_center': True,
    #     'random_size': False,
    # },
    # 'RandSpatialCropSamples': {
    #     'roi_size': None,
    #     'num_samples': 10,
    #     'max_roi_size': (100, 100, 100),
    #     'random_center': True,
    #     'random_size': False,
    # }
    # 'RandBiasField': {'degree': 3, 'coeff_range': (0.2, 0.3), 'prob': 1.0},
    # 'RandRotate90': {'prob': 0.1, 'max_k': 3, 'spatial_axes': (0, 1)},
    # 'RandGibbsNoise': {'prob': 1.0, 'alpha': (0.6, 0.8), 'as_tensor_output': True},
    # 'RandKSpaceSpikeNoise': {'prob': 1.0, 'intensity_range': (10, 13), 'channel_wise': True, 'as_tensor_output':
    # True},
    'RandCoarseDropout': {
        'holes': 200,
        'spatial_size': 10,
        'dropout_holes': True,
        'fill_value': 0,
        'max_holes': None,
        'max_spatial_size': None,
        'prob': 0.1,
    }
}
