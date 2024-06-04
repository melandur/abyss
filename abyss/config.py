import os
from collections import OrderedDict


class ConfigFile:

    def __init__(self) -> None:
        self.project_name = 'aby'
        self.experiment_name = 'train'
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
                'label_classes': OrderedDict({'background': [0], 'wt': [1, 2, 3], 'tc': [2, 3], 'en': [3]}),
                'patch_size': [128, 128, 128],
                'tta': False,
            },
            'training': {
                'fold': 0,
                'fast_dev_run': False,
                'batch_size': 2,
                'accumulate_grad_batches': 1,
                'clip_grad': {'norm': 'norm', 'value': 12},
                'num_workers': 8,
                'max_epochs': 1000,
                'learning_rate': 0.01,
                'local_rank': 0,
                'cache_rate': 0.01,
                'check_val_every_n_epoch': 1,
                'warmup_steps': 1000,
                'multi_gpu': False,
                'deterministic': False,
                'checkpoint_path': None,
                'seed': 42,  # find the truth in randomness
            },
        }
