import os
import sys
import json

from src.utilities.utils import NestedDefaultDict

from src.utilities.conf_helpers import \
    check_and_create_folder_structure,\
    check_image_search_tag_redundancy, \
    check_image_search_tag_uniqueness


class ConfigManager:
    """The pipelines control center, most parameters can be found here"""

    def __init__(self, load_conf_file_path=None):
        if not load_conf_file_path:

            project_name = 'BratsExp1'
            experiment_name = 'test'
            project_base_path = r'C:\Users\melandur\Desktop\mo\my_test'
            dataset_folder_path = r'C:\Users\melandur\Desktop\MICCAI_BraTS_2019_Data_Training\MICCAI_BraTS_2019_Data_Training\HGG'

            self.params = {
                'logger': {'level': 'INFO'},  # 'TRACE', 'DEBUG', 'INFO'

                'pipeline_steps': {
                    'dataset': True,
                    'pre_processing': False,
                    'training': False,
                    'post_processing': False
                },

                'project': {
                    'name': project_name,
                    'experiment_name': experiment_name,
                    'base_path': project_base_path,
                    'structured_dataset_store_path': os.path.join(project_base_path, project_name, experiment_name, 'structured_dataset'),
                    'preprocessed_dataset_store_path': os.path.join(project_base_path, project_name, experiment_name, 'pre_processed_dataset'),
                    'trainset_store_path': os.path.join(project_base_path, project_name, experiment_name, 'trainset'),
                    'result_store_path': os.path.join(project_base_path, project_name, experiment_name, 'results'),
                    'augmentation_store_path': os.path.join(project_base_path, project_name, experiment_name, 'aug_plots'),
                },

                'dataset': {
                    'folder_path': dataset_folder_path,
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
                    'structured_dataset_paths': NestedDefaultDict(),
                    'preprocessed_dataset_paths': NestedDefaultDict(),
                    'train_dataset_paths': NestedDefaultDict(),
                    'val_dataset_paths': NestedDefaultDict(),
                    'test_dataset_paths': NestedDefaultDict(),
                }
            }
        else:
            self.params = json.load(load_conf_file_path)

        check_image_search_tag_redundancy(self.params)
        check_image_search_tag_uniqueness(self.params)
        check_and_create_folder_structure(self.params)


if __name__ == '__main__':
    cm = ConfigManager()

