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
                'lr_decay': False,  # todo: check this
                'tta': False,
            },  # 360
            'training': {
                'fold': 0,
                'fast_dev_run': False,
                'batch_size': 8,
                'accumulate_grad_batches': 1,
                'clip_grad': {'norm': 'norm', 'value': 12.0},
                'num_workers': 8,
                'max_epochs': 1000,
                'learning_rate': 1e-2,
                'early_stop': {'patience': 50, 'min_delta': 1e-5, 'mode': 'min', 'min_lr': 1e-6},
                'local_rank': 0,
                'cache_rate': 0.01,
                'check_val_every_n_epoch': 1,
                'multi_gpu': False,
                'compile': False,  # todo: check this
                'deterministic': False,
                'checkpoint_path': None,
                'seed': 42,  # find the truth in randomness
            },
        }
