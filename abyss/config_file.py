import os


class ConfigFile:
    """The pipelines control center, all parameters can be found here"""

    def __init__(self) -> None:
        self.project_name = 'aby'
        self.experiment_name = '3'
        self.project_base_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        self.dataset_folder_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'training_abyss')

    def __call__(self) -> dict:
        """Returns config file"""
        experiment_path = os.path.join(self.project_base_path, self.project_name, self.experiment_name)
        return {
            'logger': {'level': 'INFO'},  # 'TRACE', 'DEBUG', 'INFO'
            'pipeline_steps': {
                'data_reader': True,
                'pre_processing': True,
                'create_trainset': False,
                'training': {'fit': False, 'test': False},
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
                            't1c': '_t1c.nii.gz',
                            't1': '_t1.nii.gz',
                            't2': '_t2.nii.gz',
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
                        'resize': {'active': True, 'dim': (128, 128, 128), 'interpolator': 'sitk.sitkNearestNeighbor'},
                        'relabel_mask': {'active': False, 'label_dict': {1: 2, 2: 1, 3: 10}},  # original : new
                        'simple_itk_writer': {'active': True, 'file_type': 'sitk.sitkInt8'},
                    },
                },
                'data': {
                    'images': {
                        'simple_itk_reader': {'active': True, 'orientation': 'LPS', 'file_type': 'sitk.sitkFloat32'},
                        'crop_zeros': {'active': True},
                        'resize': {'active': True, 'dim': (128, 128, 128), 'interpolator': 'sitk.sitkLinear'},
                        'clip_percentiles': {'active': True, 'lower': 0.1, 'upper': 99.9},
                        'z_score_norm': {'active': True},
                        'simple_itk_writer': {'active': True, 'file_type': 'sitk.sitkFloat32'},
                    },
                },
            },
            'training': {
                'batch_size': 80,
                'optimizers': {
                    'Adam': {
                        'active': True,
                        'learning_rate': 0.001,
                        'betas': (0.9, 0.999),
                        'eps': 1e-8,
                        'weight_decay': 1e-2,
                        'amsgrad': False,
                    },
                    'SGD': {
                        'active': True,
                        'learning_rate': 0.01,
                        'momentum': 0.9,
                        'weight_decay': 1e-2,
                        'nesterov': True,
                    },
                },
                'criterion': 'dice',  # mse, cross_entropy, dice, cross_entropy_dice
                'load_from_checkpoint_path': None,  # loads if valid *.ckpt provided
                'load_from_weights_path': None,  # loads if valid *.pth provided
            },
            'trainer': {
                'default_root_dir': os.path.join(experiment_path, '4_results'),
                'max_epochs': 1000,
                'log_every_n_steps': 1,
                'precision': 32,
                'check_val_every_n_epoch': 1,
                'enable_progress_bar': True,
                'accelerator': 'gpu',
                'deterministic': False,
                'devices': 1,
                'gpus': None,
                'resume_from_checkpoint': None,
                'model_summary_depth': -1,
                'early_stop': {'min_delta': 0.01, 'patience': 5, 'verbose': False, 'mode': 'max'},
            },
            'production': {
                'checkpoint_name': None,
                'weights_name': f'{self.project_name}_{self.experiment_name}.pth',
            },
        }

    def __str__(self) -> str:
        return self.__class__.__name__
