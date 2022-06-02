import os


class ConfigFile:
    """The pipelines control center, all parameters can be found here"""

    def __init__(self):
        self.project_name = 'Abyss_test'
        self.experiment_name = 'experiment_2'
        self.project_base_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        self.dataset_folder_path = '/home/melandur/Data/small'

    def __call__(self):
        """Returns config file"""
        experiment_path = os.path.join(self.project_base_path, self.project_name, self.experiment_name)
        return {
            'logger': {'level': 'TRACE'},  # 'TRACE', 'DEBUG', 'INFO'
            'pipeline_steps': {
                'data_reader': False,
                'pre_processing': True,
                'create_trainset': False,
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
                'label_search_tags': ['seg', 'Seg'],
                'get_case_name_from': 'case_folder',
                'image_file_type': ['.nii.gz'],
                'image_search_tags': {
                    't1': ['t1.'],
                    't1c': ['t1ce.'],
                    'flair': ['flair.'],
                    't2': ['t2.'],
                },
                'val_frac': 0.2,
                'test_frac': 0.2,
            },
            'pre_processing': {
                'image': {
                    'canonical': {'active': True},
                    'resize': {
                        'active': True,
                        'dim': (100, 100, 100),
                        'interpolator': 'linear'},
                    'z_score': {'active': True},
                    'rescale_intensity': {'active': True}
                },
                'label': {
                    'canonical': {'active': True},
                    'resize': {
                        'active': True,
                        'dim': (100, 100, 100),
                        'interpolator': 'nearest'}
                }
            },
            'augmentation': {},
            'training': {
                'epochs': 30,  # tbd
                'trained_epochs': None,
                'batch_size': 1,  # tbd
                'optimizer': 'Adam',  # Adam, SGD
                'learning_rate': 1e-3,  # tbd
                'betas': (0.9, 0.999),  # tbd
                'eps': 1e-8,
                'weight_decay': 1e-5,  # tbd
                'amsgrad': True,
                'dropout': 0.5,  # tbd
                'criterion': ['MSE_mean'],
                'num_workers': 8,
                'n_classes': 3,
                'early_stop': {'min_delta': 0.0, 'patience': 0, 'verbose': False, 'mode': 'max'},
            },
            'post_processing': {},
        }

    def __str__(self):
        return self.__class__.__name__
