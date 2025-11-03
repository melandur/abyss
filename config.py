import os


class ConfigFile:

    def __init__(self) -> None:
        self.project_name = 'aby'
        self.experiment_name = 'gbm'
        self.project_path = os.path.join(os.path.expanduser('~'), 'code', 'abyss', 'data')
        self.train_dataset_path = os.path.join(os.path.expanduser('~'), 'code', 'abyss', 'data', 'training', 'gbm')
        self.test_dataset_path = os.path.join(os.path.expanduser('~'), 'code', 'abyss', 'data', 'training', 'raw')
        self.pretrained_checkpoint_path = os.path.join(
            os.path.expanduser('~'), 'code', 'abyss', 'transfer', 'checkpoint_final_ResEncL_MAE.pth'
        )  # Path to checkpoint, None to disable
        self.pretrained_checkpoint_path = None

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
                'channel_order': {'t1c': '_t1c.nii.gz'},
                'label_order': {'seg': '_seg.nii.gz'},
                'seed': 42,
            },
            'mode': {'train': True, 'test': False},
            'trainer': {
                'label_classes': {'en': [2]},
                'patch_size': [160, 160, 160],  # y, x, z
                'task': 'segmentation',  # 'segmentation', 'classification', 'detection'
            },
            'training': {
                'fold': 0,
                'fast_dev_run': False,
                'num_workers': 12,
                'local_rank': 0,
                'cache_rate': 0.1,
                'compile': False,
                'multi_gpu': False,
                'deterministic': False,
                'reload_checkpoint': False,
                'checkpoint_path': None,
                'batch_size': 2,
                'epochs': 1000,
                'warmup_epochs': 0,
                'lr': 1e-2,
                'seed': 42,  # find the truth in randomness
                'pretrained_checkpoint_path': self.pretrained_checkpoint_path,
                'loss': 'dice_ce',  # 'ce' | 'dice_ce' | 'dice_ce_topk'
            },
        }
