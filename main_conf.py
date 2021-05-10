import sys
import json


class ConfigManager:
    """The pipelines control center, most parameters can be found here"""

    def __init__(self, load_conf_file_path=None):
        if not load_conf_file_path:
            self.params = {
                'logger': {'level': 'INFO'},  # 'TRACE', 'DEBUG', 'INFO', 'ERROR'

                'pipeline_steps': {'dataset': False,
                                   'pre_processing': False,
                                   'training': True,
                                   'post_processing': True
                                   },

                'project': {'name': 'BratsExp1',
                            'dataset_store_path': r'C:\Users\melandur\Desktop\mo\my_test',
                            'result_store_path': r'C:\Users\melandur\Desktop\mo\logs',
                            'augmentation_store_path': r'C:\Users\melandur\Desktop\mo\my_test\aug',
                            },

                'dataset': {
                    'folder_path': r'C:\Users\melandur\Desktop\MICCAI_BraTS_2019_Data_Training\MICCAI_BraTS_2019_Data_Training\HGG',
                    'label_search_tags': ['seg.'],
                    'label_file_type': ['.nii.gz'],
                    'image_search_tags': {'t1': ['t1.'],
                                          't1ce': ['t1ce.'],
                                          'flair': ['flair.'],
                                          't2': ['t2.']},
                    'image_file_type': ['.nii.gz'],

                    'pull_dataset': 'DecathlonDataset',  # 'MedNISTDataset', 'DecathlonDataset', 'CrossValidation'
                    'challenge': 'Task01_BrainTumour',
                    # only need for decathlon:   'Task01_BrainTumour', 'Task02_Heart', 'Task03_Liver0', 'Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon'
                    'seed': 42,
                    'val_frac': 0.2,
                    'test_frac': 0.2,
                    'use_cache': False,  # goes heavy on memory
                    'cache_max': sys.maxsize,
                    'cache_rate': 0.0,  # set 0 to reduce memory consumption
                    'num_workers': 8
                },

                'pre_processing': {},
                'post_processing': {},

                'augmentation': {},

                'training': {
                    'seed': 42,
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
                    'n_classes': 3
                },

                'tmp': {
                    'data_path_store': dict,
                    'train_data_path_store': dict,
                    'val_data_path_store': dict,
                    'test_data_path_store': dict,
                }

            }
        else:
            self.params = json.load(load_conf_file_path)
