import os


class ConfigFile:
    """The pipelines control center, all parameters can be found here"""

    def __init__(self):
        self.project_name = 'aby'
        self.experiment_name = '1'
        self.project_base_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        self.dataset_folder_path = '/home/melandur/Data/small_train'

    def __call__(self) -> dict:
        """Returns config file"""
        experiment_path = os.path.join(self.project_base_path, self.project_name, self.experiment_name)
        return {
            'logger': {'level': 'INFO'},  # 'TRACE', 'DEBUG', 'INFO'
            'pipeline_steps': {
                'data_reader': False,
                'pre_processing': False,
                'create_trainset': False,
                'training': False,
                'production': True,
                'inference': True,
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
                'production_store_path': os.path.join(experiment_path, '5_production'),
                'inference_store_path': os.path.join(experiment_path, '6_inference'),
                'postprocessed_store_path': os.path.join(experiment_path, '7_post_processed'),
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
                'val_fraction': 0.2,  # only used when cross_fold = 1/1, otherwise defined as 1/max_number_of_folds
                'test_fraction': 0.2,
                'cross_fold': '1/1',
            },
            'pre_processing': {
                'data': {
                    'orient_to_ras': {'active': True},
                    'resize': {'active': True, 'dim': (128, 128, 128), 'interpolator': 'linear'},
                    'z_score': {'active': True},
                    'rescale_intensity': {'active': False},
                },
                'label': {
                    'orient_to_ras': {'active': True},
                    'resize': {'active': True, 'dim': (128, 128, 128), 'interpolator': 'nearest'},
                    'remap_labels': {'active': True, 'label_dict': {1: 1, 2: 2, 4: 3}},  # original : new
                },
            },
            'training': {
                'batch_size': 1,
                'optimizers': {
                    'Adam': {
                        'active': False,
                        'learning_rate': 0.001,
                        'betas': (0.9, 0.999),
                        'eps': 1e-8,
                        'weight_decay': 0,
                        'amsgrad': False,
                    },
                    'SGD': {
                        'active': True,
                        'learning_rate': 0.01,
                        'momentum': 0.9,
                        'weight_decay': 0.99,
                        'nesterov': True,
                    },
                },
                'criterion': 'cross_entropy_dice',  # mse, cross_entropy, dice, cross_entropy_dice
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
                'stochastic_weight_avg': False,
                'accelerator': 'gpu',
                'deterministic': False,
                'devices': 1,
                'gpus': None,
                'auto_select_gpus': False,
                'tpu_cores': None,
                'fast_dev_run': False,
                'resume_from_checkpoint': None,
                'model_summary_depth': -1,
                'early_stop': {'min_delta': 0.01, 'patience': 5, 'verbose': False, 'mode': 'max'},
            },
            'production': {'checkpoint_path': None},
            'inference': {'weights_path': None},
        }

    def __str__(self) -> str:
        return self.__class__.__name__
