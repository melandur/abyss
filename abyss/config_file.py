import os


class ConfigFile:
    """The pipelines control center, all parameters can be found here"""

    def __init__(self):
        self.project_name = 'Abyss_test'  # tbd
        self.experiment_name = 'experiment_3'  # tbd
        self.project_base_path = os.path.join(os.path.expanduser('~'), 'Downloads')  # tbd
        self.dataset_folder_path = '/home/melandur/Data/small'  # tbd

    def __call__(self) -> dict:
        """Returns config file"""
        experiment_path = os.path.join(self.project_base_path, self.project_name, self.experiment_name)
        return {
            'logger': {'level': 'INFO'},  # 'TRACE', 'DEBUG', 'INFO'
            'pipeline_steps': {
                'data_reader': True,
                'pre_processing': True,
                'create_trainset': True,
                'training': False,
                'post_processing': False,
            },
            'project': {
                'name': self.project_name,
                'experiment_name': self.experiment_name,
                'base_path': self.project_base_path,
                'dataset_folder_path': self.dataset_folder_path,
                'config_store_path': os.path.join(experiment_path, '0_config_data'),
                'structured_dataset_store_path': os.path.join(experiment_path, '1_structured_dataset'),
                'preprocessed_dataset_store_path': os.path.join(experiment_path, '2_pre_processed_dataset'),
                'trainset_store_path': os.path.join(experiment_path, '3_trainset'),
                'result_store_path': os.path.join(experiment_path, '4_results'),
            },
            'meta': {
                'seed': 42,  # find the truth in randomness
                'num_workers': 8,
            },
            'dataset': {
                'folder_path': self.dataset_folder_path,
                'label_file_type': ['.nii.gz'],
                'label_search_tags': {'mask': ['seg', 'Seg']},
                'data_file_type': ['.nii.gz'],
                'data_search_tags': {
                    'flair': ['flair.', 'FLAIR.'],
                    't1c': ['t1ce.', 'T1CE.'],
                    't2': ['t2.', 'T2.'],
                    't1': ['t1.', 'T1.'],
                },
                'val_fraction': 0.2,  # only used when cross_fold = 1/1
                'test_fraction': 0.2,
                'cross_fold': '1/1',
            },
            'pre_processing': {
                'data': {
                    'orient_to_ras': {'active': True},
                    'resize': {'active': True, 'dim': (100, 100, 100), 'interpolator': 'linear'},
                    'z_score': {'active': True},
                    'rescale_intensity': {'active': True},
                },
                'label': {
                    'orient_to_ras': {'active': True},
                    'resize': {'active': True, 'dim': (100, 100, 100), 'interpolator': 'nearest'},
                },
            },
            'augmentation': {
                # based on monai.transforms | comment/delete unused ones | order corresponds to the application sequence
                # https://docs.monai.io/en/stable/transforms.html#vanilla-transforms
                'RandGaussianNoise': {'prob': 0.1, 'mean': 0.0, 'std': 0.1},
                'RandGaussianSmooth': {
                    'sigma_x': (0.25, 1.5),
                    'sigma_y': (0.25, 1.5),
                    'sigma_z': (0.25, 1.5),
                    'prob': 0.1,
                    'approx': 'erf',
                },
                'RandScaleIntensity': {'factors': (1.0, 1.0), 'prob': 0.1},
                'RandFlip': {'prob': 0.1, 'spatial_axis': None},
                # 'RandAdjustContrast': {'prob': 0.1, 'gamma': (0.5, 4.5)},
                'RandRotate': {
                    'range_x': 0.0,
                    'range_y': 0.0,
                    'range_z': 0.0,
                    'prob': 0.1,
                    'keep_size': True,
                    'mode': 'bilinear',
                    'padding_mode': 'border',
                    'align_corners': False,
                },
                # 'RandScaleCrop': {
                #     'roi_scale': [1.0, 1.0],
                #     'max_roi_scale': None,
                #     'random_center': True,
                #     'random_size': True,
                # },
                # 'RandHistogramShift': {'num_control_points': 10, 'prob': 0.1},
                # 'RandSpatialCrop': {
                #     'roi_size': [1, 1],
                #     'max_roi_size': None,
                #     'random_center': True,
                #     'random_size': True,
                # },
                # 'RandBiasField': {'degree': 3, 'coeff_range': (0.0, 0.1), 'prob': 0.1},
                # 'Rand2DElastic': {
                #     'spacing': 1.0,
                #     'magnitude_range': (1.0, 1.0),
                #     'prob': 0.1,
                #     'rotate_range': None,
                #     'shear_range': None,
                #     'translate_range': None,
                #     'scale_range': None,
                #     'spatial_size': None,
                #     'mode': 'bilinear',
                #     'padding_mode': 'reflection',
                #     'as_tensor_output': False,
                #     'device': None,
                # },
                # 'Rand3DElastic': {
                #     'sigma_range': (1.0, 1.0),
                #     'magnitude_range': (1.0, 1.0),
                #     'prob': 0.1,
                #     'rotate_range': None,
                #     'shear_range': None,
                #     'translate_range': None,
                #     'scale_range': None,
                #     'spatial_size': None,
                #     'mode': 'bilinear',
                #     'padding_mode': 'reflection',
                #     'as_tensor_output': False,
                #     'device': None,
                # },
                # 'RandAffine': {
                #     'prob': 0.1,
                #     'rotate_range': None,
                #     'shear_range': None,
                #     'translate_range': None,
                #     'scale_range': None,
                #     'spatial_size': None,
                #     'mode': 'bilinear',
                #     'padding_mode': 'reflection',
                #     'cache_grid': False,
                #     'as_tensor_output': True,
                #     'device': None,
                # },
                'RandRotate90': {'prob': 0.1, 'max_k': 3, 'spatial_axes': (0, 1)},
            },
            'training': {
                'batch_size': 1,  # tbd
                'optimizer': 'Adam',  # Adam, SGD
                'learning_rate': 1e-3,  # tbd
                'betas': (0.9, 0.999),  # tbd
                'eps': 1e-8,
                'weight_decay': 1e-5,  # tbd
                'amsgrad': True,
                'dropout': 0.5,  # tbd
                'criterion': ['MSE_mean'],
            },
            'trainer': {
                'default_root_dir': os.path.join(experiment_path, '4_results'),
                'max_epochs': 1000,
                'log_every_n_steps': 1,
                'precision': 32,
                'check_val_every_n_epoch': 1,
                'enable_progress_bar': True,
                'stochastic_weight_avg': False,
                'accelerator': None,
                'deterministic': True,
                'devices': None,
                'gpus': None,
                'auto_select_gpus': False,
                'tpu_cores': None,
                'fast_dev_run': False,
                'resume_from_checkpoint': None,
                'auto_lr_find': False,
                'model_summary_depth': 3,
                'early_stop': {'min_delta': 0.01, 'patience': 5, 'verbose': False, 'mode': 'max'},
            },
            'post_processing': {},
        }

    def __str__(self) -> str:
        return self.__class__.__name__
