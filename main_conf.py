import os
import sys
import json
from copy import deepcopy
from collections import Counter
from loguru import logger as log


class ConfigManager:
    """The pipelines control center, most parameters can be found here"""

    def __init__(self, load_conf_file_path=None):
        if not load_conf_file_path:
            self.params = {
                'logger': {'level': 'INFO'},  # 'TRACE', 'DEBUG', 'INFO', 'ERROR'

                'pipeline_steps': {
                    'dataset': True,
                    'pre_processing': False,
                    'training': False,
                    'post_processing': False
                },

                'project': {
                    'name': 'BratsExp1',
                    'dataset_store_path': r'C:\Users\melandur\Desktop\mo\my_test',
                    'result_store_path': r'C:\Users\melandur\Desktop\mo\logs',
                    'augmentation_store_path': r'C:\Users\melandur\Desktop\mo\my_test\aug',
                },

                'dataset': {
                    'folder_path': r'C:\Users\melandur\Desktop\MICCAI_BraTS_2019_Data_Training\MICCAI_BraTS_2019_Data_Training\HGG',
                    'label_search_tags': ['seg.'],
                    'label_file_type': ['.nii.gz'],
                    'image_search_tags': {
                        't1': ['t1.'],
                        't1ce': ['t1ce.'],
                        'flair': ['flair.'],
                        't2': ['t2']},
                    'image_file_type': ['.nii.gz'],
                    'concatenate_image_files': True,
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
                    'n_classes': 3,
                    'early_stop': {
                        'min_delta': 0.0,
                        'patience': 0,
                        'verbose': False,
                        'mode': 'max'
                    }
                },

                'tmp': {
                    'data_path_store': dict,
                    'train_data_path_store': dict,
                    'val_data_path_store': dict,
                    'test_data_path_store': dict,
                    'copy_manager_': dict,
                }
            }
        else:
            self.params = json.load(load_conf_file_path)

        self.__check_image_search_tag_redundancy()
        self.__check_image_search_tag_uniqueness()
        self.__check_and_create_folder_structure()

    def __check_and_create_folder_structure(self):
        """Check and create folders if they are missing"""
        folders = [
            self.params['project']['dataset_store_path'],
            self.params['project']['result_store_path'],
            self.params['project']['augmentation_store_path'],
            self.params['dataset']['folder_path']
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def __check_image_search_tag_redundancy(self):
        """Check if there are any redundant search tag per image name"""
        for key, value in self.params['dataset']['image_search_tags'].items():
            if len(value) != len(set(value)):
                redundant_tag = list((Counter(value) - Counter(list(set(value)))).elements())
                log.error(f'The image search tag {redundant_tag} appears multiple times for the image name {key}')
                exit(1)

    def __check_image_search_tag_uniqueness(self):
        """Check if the image search tags are unique enough to avoid wrong data loading"""
        tags = [*self.params['dataset']['image_search_tags'].values()]
        tags = [x for sublist in tags for x in sublist]  # flatten nested list
        for i, tag in enumerate(tags):
            tmp_tags = deepcopy(tags)
            tmp_tags.pop(i)
            if [x for x in tmp_tags if x in tag]:
                vague_tag = [x for x in tmp_tags if x in tag]
                log.error(f'The image search tag {vague_tag} is not expressive/unique enough. '
                          f'Try to add additional information to the search tag like "_", "."')
                exit(1)


if __name__ == '__main__':
    cm = ConfigManager()

