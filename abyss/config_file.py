import os


class ConfigFile:
    """The pipelines control center, all parameters can be found here"""

    def __init__(self) -> None:
        self.project_name = 'aby'
        self.experiment_name = 'brats'
        self.project_base_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        self.dataset_folder_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'MICCAI_4')

    def __call__(self) -> dict:
        """Returns config file"""
        experiment_path = os.path.join(self.project_base_path, self.project_name, self.experiment_name)
        return {
            'logger': {'level': 'INFO'},  # 'TRACE', 'DEBUG', 'INFO'
            'pipeline_steps': {
                'data_reader': False,
                'pre_processing': False,
                'create_trainset': False,
                'training': {'fit': True, 'test': False},
                'production': {'extract_weights': False, 'inference': False, 'post_processing': False},
            },
            'project': {
                'name': self.project_name,
                'experiment_name': self.experiment_name,
                'base_path': self.project_base_path,
                'dataset_folder_path': self.dataset_folder_path,
                'config_store_path': os.path.join(experiment_path, '0_config_data'),
                'structured_dataset_store_path': os.path.join(experiment_path, '1_structured_dataset'),
                'pre_processed_dataset_store_path': os.path.join(experiment_path, '2_pre_processed_dataset'),
                'trainset_store_path': os.path.join(experiment_path, '3_trainset'),
                'result_store_path': os.path.join(experiment_path, '4_results'),
                'production_store_path': os.path.join(experiment_path, '5_production'),
            },
            'meta': {
                'seed': 42,  # find the truth in randomness
                'num_workers': 8,
            },
            'dataset': {
                'folder_path': self.dataset_folder_path,
                'description': {
                    'labels': {
                        'images': {'mask': '_seg.nii.gz'},
                    },
                    'data': {
                        'images': {
                            't1c': '_t1ce.nii.gz',
                            't1': '_t1.nii.gz',
                            't2': '_t2.nii.gz',
                            'flair': '_flair.nii.gz',
                        },
                    },
                },
                'val_fraction': 0.2,  # only used when cross_fold = 1/1, otherwise defined as 1/max_number_of_folds
                'test_fraction': 0.2,  # todo: goes trainer i guess
                'cross_fold': '1/1',
            },
            'pre_processing': {
                'labels': {
                    'images': {
                        'simple_itk_reader': {'active': True, 'orientation': 'LPS', 'file_type': 'sitk.sitkInt8'},
                        'crop_zeros': {'active': True},
                        'resize_image': {
                            'active': True,
                            'dim': (240, 240, 150),
                            'interpolator': 'nearest',
                        },
                        'relabel_mask': {'active': False, 'label_dict': {1: 1, 2: 2, 3: 3, 4: 3}},  # original : new
                        'simple_itk_writer': {'active': True, 'file_type': 'sitk.sitkInt8'},
                    },
                },
                'data': {
                    'images': {
                        'simple_itk_reader': {'active': True, 'orientation': 'LPS', 'file_type': 'sitk.sitkFloat32'},
                        'background_as_zeros': {'active': True, 'threshold': 0},
                        'crop_zeros': {'active': True},
                        'resize_image': {'active': True, 'dim': (240, 240, 150), 'interpolator': 'trilinear'},
                        'clip_percentiles': {'active': True, 'lower': 0.5, 'upper': 99.5},
                        'z_score_norm': {'active': True, 'foreground_only': True},
                        'simple_itk_writer': {'active': True, 'file_type': 'sitk.sitkFloat32'},
                    },
                },
            },
            'training': {
                'output_channels': {
                    'enhanced': 4,
                    'necrosis': 1,
                    'edema': 2,
                },
                'batch_size': 8,
                'optimizers': {
                    'Adam': {
                        'active': False,
                        'learning_rate': 3e-4,
                        'betas': (0.9, 0.999),
                        'eps': 1e-3,
                        'weight_decay': 3e-5,
                        'amsgrad': True,
                    },
                    'SGD': {
                        'active': True,
                        'learning_rate': 1e-3,
                        'momentum': 0.99,
                        'weight_decay': 3e-5,
                        'nesterov': True,
                    },
                },
                'criterion': 'cross_entropy_dice',  # mse, cross_entropy, dice, cross_entropy_dice
                'load_from_checkpoint_path': None,  # loads if valid *.ckpt provided
                'load_from_weights_path': None,  # loads if valid *.pth provided
            },
            'trainer': {
                'default_root_dir': os.path.join(experiment_path, '4_results'),
                'total_epochs': 1000,
                'automated_mixed_precision': {  # todo: implement
                    'active': True,
                    'device': 'cuda',
                    'precision': 'float16',  # 'bfloat32', 'float16'
                },
                'val_epoch': 10,
                'auto_save': {'active': False, 'n_epoch': 1000},
                'accelerator': 'gpu',
                'deterministic': False,
                'compile': False,
                'resume_from_checkpoint': None,
                'early_stop': {'patience': 17, 'min_learning_rate': 1e-6, 'min_delta': 1e-4},
                'lr_scheduler': {'warmup_epochs': 50},
                'metrics': ['dice_score'],
            },
            'production': {
                'checkpoint_name': None,
                'weights_name': f'{self.project_name}_{self.experiment_name}.pth',
            },
        }

    def __str__(self) -> str:
        return self.__class__.__name__
