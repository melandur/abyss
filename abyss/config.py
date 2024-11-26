import os
from collections import OrderedDict



class ConfigFile:

    def __init__(self) -> None:
        self.project_name = 'aby'
        self.experiment_name = 'mets'
        self.project_path = os.path.join(os.path.expanduser('~'), 'code', 'abyss', 'data')
        self.train_dataset_path = os.path.join(os.path.expanduser('~'), 'code', 'abyss', 'data', 'data', 'mets')
        self.test_dataset_path = os.path.join(os.path.expanduser('~'), 'code', 'abyss', 'data', 'raw')

    def get_config(self) -> dict:
        """Returns config dict"""
        experiment_path = os.path.join(self.project_path, self.project_name, self.experiment_name)
        return {
            'logger': {'level': 'INFO'},  # 'TRACE', 'DEBUG', 'INFO'
            'project': {
                'name': self.project_name,
                'experiment_name': self.experiment_name,
                'base_path': self.project_path,
                'train_dataset_path': self.train_dataset_path,
                'test_dataset_path': self.test_dataset_path,
                'config_path': os.path.join(experiment_path, '0_config'),
                'results_path': os.path.join(experiment_path, '1_results'),
                'inference_path': os.path.join(experiment_path, '2_inference'),
            },
            'dataset': {
                'spacing': [1.0, 1.0, 1.0],
                'total_folds': 5,
                # 'channel_order': {
                    # 't1': '_t1.nii.gz',
                    # 't1c': '_t1c.nii.gz'
                    # 't2': '_t2.nii.gz',
                    # 'flair': '_flair.nii.gz',
                # },
                'channel_order': {'t1c': '_t1c.nii.gz'},
                'label_order': {'seg': '_seg.nii.gz'},
                'seed': 42,
            },
            'mode': {'train': True, 'test': False},
            'trainer': {
                # 'label_classes': OrderedDict({'wt': [1, 2, 3, 4], 'tc': [2, 3, 4], 'en': [2]}),
                'label_classes': OrderedDict({'wt': [1, 2], 'en': [1]}),
                'patch_size': [128, 128, 128],  # y, x, z
            },
            'training': {
                'fold': 0,
                'fast_dev_run': False,
                'num_workers': 12,
                'local_rank': 0,
                'cache_rate': 0.1,
                'compile': True,
                'multi_gpu': False,
                'deterministic': True,
                'reload_checkpoint': False,
                'checkpoint_path': None,
                'seed': 42,  # find the truth in randomness
            },
        }
