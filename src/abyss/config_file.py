import os


class ConfigFile:
    """The pipelines control center, all parameters can be found here"""

    def __init__(self) -> None:
        self.project_name = 'abyss_test'
        self.experiment_name = '1'
        self.run_name = 'run_1'
        self.project_base_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        self.dataset_folder_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'kitten_11')

    def __call__(self) -> dict:
        """Returns config file"""
        experiment_path = os.path.join(self.project_base_path, self.project_name, self.experiment_name)
        return {
            'logger': 'INFO',  # 'TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
            'pipeline_steps': {
                'data_reader': False,
                'pre_processing': False,
                'create_trainset': False,
                'training': {
                    'fit': True,
                    'test': False,
                },
                'production': {
                    'extract_weights': False,
                    'inference': False,
                    'post_processing': False,
                },
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
                'train_log_store_path': os.path.join(experiment_path, '4_results', 'train_logs', self.run_name),
                'checkpoint_store_path': os.path.join(experiment_path, '4_results', 'checkpoints', self.run_name),
                'production_store_path': os.path.join(experiment_path, '5_production'),
            },
            'meta': {
                'seed': 42,  # find the truth in randomness
                'num_workers': 8,
            },
            'dataset': {
                'folder_path': self.dataset_folder_path,
                'label_file_type': ['.nii.gz'],
                'label_search_tags': {
                    'mask': ['_seg.'],
                },
                'data_file_type': ['.nii.gz'],
                'data_search_tags': {
                    't1': ['_t1.'],
                    't1c': ['_t1c.'],
                    't2': ['_t2.'],
                    'flair': ['_flair.'],
                },
                'val_fraction': 0.2,  # only used when cross_fold = 1/1, otherwise defined as 1/max_number_of_folds
                'test_fraction': 0.2,
                'cross_fold': '1/1',
            },
            'pre_processing': {
                'data': {
                    'orient_to_ras': {'active': False},
                    'resize': {
                        'active': True,
                        'dim': (128, 128, 128),
                        'interpolator': 'linear',
                    },  # (height, width, depth)
                    'z_score': {'active': True},
                    'rescale_intensity': {'active': True},
                },
                'label': {
                    'orient_to_ras': {'active': False},
                    'resize': {
                        'active': True,
                        'dim': (128, 128, 128),
                        'interpolator': 'nearest',
                    },  # (height, width, depth)
                    'remap_labels': {
                        'active': False,
                        'label_dict': {1: 1, 2: 2, 3: 3},
                    },  # original : new
                },
            },
            'training': {
                'batch_size': 9,
                'optimizers': {
                    'Adam': {
                        'active': True,
                        'learning_rate': 0.001,
                        'betas': (0.9, 0.999),
                        'eps': 1e-8,
                        'weight_decay': 1e-2,
                        'amsgrad': False,
                    },
                    # 'SGD': {
                    #     'active': True,
                    #     'learning_rate': 0.01,
                    #     'momentum': 0.9,
                    #     'weight_decay': 1e-2,
                    #     'nesterov': True,
                    # },
                },
                'criterion': 'dice',  # mse, cross_entropy, dice, cross_entropy_dice
                'log_metrics': ['dice'],  # dice, accuracy
                'load_from_checkpoint_path': None,  # loads if valid *.ckpt provided
                'load_from_weights_path': None,  # loads if valid *.pth provided
            },
            'trainer': {
                'default_root_dir': os.path.join(experiment_path, '4_results'),
                'max_epochs': 1000,
                'log_every_n_steps': 10,
                'precision': 32,
                'check_val_every_n_epoch': 1,
                'enable_progress_bar': True,
                'accelerator': 'gpu',
                'deterministic': False,
                'devices': 1,
                'gpus': None,
                'tpu_cores': None,
                'fast_dev_run': False,
                'resume_from_checkpoint': None,
                'model_summary_depth': -1,
                'early_stop': {
                    'monitor': 'val_dice',
                    'min_delta': 0.01,
                    'patience': 5,
                    'verbose': False,
                    'mode': 'max',
                },
            },
            'production': {
                'checkpoint_name': None,
                'weights_name': f'{self.project_name}_{self.experiment_name}.pth',
            },
        }

    def __str__(self) -> str:
        return self.__class__.__name__
