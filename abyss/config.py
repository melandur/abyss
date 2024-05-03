import os


class ConfigFile:

    def __init__(self) -> None:
        self.project_name = 'aby'
        self.experiment_name = 'brats'
        self.project_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        self.dataset_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'Task01_BrainTumour')

    def get_config(self) -> dict:
        """Returns config dict"""
        experiment_path = os.path.join(self.project_path, self.project_name, self.experiment_name)
        return {
            'logger': {'level': 'INFO'},  # 'TRACE', 'DEBUG', 'INFO'
            'project': {
                'name': self.project_name,
                'experiment_name': self.experiment_name,
                'base_path': self.project_path,
                'dataset_path': self.dataset_path,
                'config_path': os.path.join(experiment_path, '0_config'),
                'results_path': os.path.join(experiment_path, '1_results'),
                'inference_path': os.path.join(experiment_path, '2_inference'),
            },
            'dataset': {
                'spacing': [1.0, 1.0, 1.0],
                'clip_values': [0, 0],
                'normalize_values': [0, 0],
                'total_folds': 5,
                'seed': 42,
            },
            'mode': {'train': True, 'test': False},
            'trainer': {
                'batch_size': 3,
                'patch_size': [128, 128, 128],
                'lr_decay': True,
                'tta': True,
            },
            'training': {
                'fold': 0,
                'batch_size': {'train': 3, 'val': 1},
                'num_workers': {'train': 4, 'val': 4},
                'max_epochs': 1000,
                'learning_rate': 1e-3,
                'warmup': {'active': True, 'epochs': 3},
                'early_stop': {'active': True, 'patience': 10},
                'local_rank': 0,
                'cache_rate': 0.1,
                'val_interval': 1,
                'multi_gpu': False,
                'amp': False,
                'compile': False,  # todo: add this to the pipeline
                'deterministic': False,
                'checkpoint_path': None,
                'seed': 42,  # find the truth in randomness
            },
        }
