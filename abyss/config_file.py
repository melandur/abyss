import os


class ConfigFile:
    """The pipelines control center, all parameters can be found here"""

    def __init__(self):
        self.project_name = 'Abyss_test'  # tbd
        self.experiment_name = 'experiment_3'  # tbd
        self.project_base_path = os.path.join(os.path.expanduser('~'), 'Downloads')  # tbd
        self.dataset_folder_path = '/home/melandur/Data/small'  # tbd

    def __call__(self):
        """Returns config file"""
        experiment_path = os.path.join(self.project_base_path, self.project_name, self.experiment_name)
        return {
            'logger': {'level': 'INFO'},  # 'TRACE', 'DEBUG', 'INFO'
            'pipeline_steps': {
                'data_reader': False,
                'pre_processing': False,
                'create_trainset': False,
                'training': True,
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
                'label_search_tags': {'net': ['seg', 'Seg']},
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
            'augmentation': {},
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
                'log_every_n_steps': 50,
                'precision': 32,
                'check_val_every_n_epoch': 1,
                'enable_progress_bar': True,
                'enable_model_summary': True,
                'weights_summary': 'top',
                'stochastic_weight_avg': False,
                'accelerator': None,
                'deterministic': None,
                'devices': None,
                'gpus': None,
                'auto_select_gpus': False,
                'tpu_cores': None,
                'fast_dev_run': False,
                'resume_from_checkpoint': None,
                'auto_lr_find': False,
                'early_stop': {'min_delta': 0.0, 'patience': 0, 'verbose': False, 'mode': 'max'},
            },
            'post_processing': {},
        }

    def __str__(self):
        return self.__class__.__name__
