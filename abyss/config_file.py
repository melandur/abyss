import os
import sys


class ConfigFile:
    """The pipelines control center, most parameters can be found here"""

    def __init__(self, project_name, experiment_name, project_base_path, dataset_folder_path):
        self.experiment_path = os.path.join(project_base_path, project_name, experiment_name)
        self.project_name = project_name
        self.project_base_path = project_base_path
        self.experiment_name = experiment_name
        self.experiment_path = os.path.join(project_base_path, experiment_name)
        self.dataset_folder_path = dataset_folder_path

    def __call__(self):
        """Returns config file"""
        return {
            'logger': {'level': 'TRACE'},  # 'TRACE', 'DEBUG', 'INFO'
            'pipeline_steps': {
                'clean_dataset': True,
                'pre_processing': False,
                'create_trainset': False,
                'training': False,
                'post_processing': False,
            },
            'project': {
                'name': self.project_name,
                'experiment_name': self.experiment_name,
                'base_path': self.project_base_path,
                'structured_dataset_store_path': os.path.join(self.experiment_path, 'structured_dataset'),
                'preprocessed_dataset_store_path': os.path.join(self.experiment_path, 'pre_processed_dataset'),
                'trainset_store_path': os.path.join(self.experiment_path, 'trainset'),
                'result_store_path': os.path.join(self.experiment_path, 'results'),
                'augmentation_store_path': os.path.join(self.experiment_path, 'aug_plots'),
                'config_store_path': os.path.join(self.experiment_path, 'config_data'),
            },
            'dataset': {
                'folder_path': self.dataset_folder_path,
                'label_search_tags': ['seg.'],
                'label_file_type': ['.nii.gz'],
                'image_search_tags': {'t1': ['T1.'],
                                      't1c': ['T1c'],
                                      'flair': ['flair.', 'Flair.'],
                                      't2': ['T2.']},
                'image_file_type': ['.nii.gz'],
                'data_reader': 'NibabelReader',
                # 'ITKReader', 'NibabelReader', 'NumpyReader', 'PILReader', 'WSIReader'
                'concatenate_image_files': True,
                'pull_dataset': 'DecathlonDataset',  # 'MedNISTDataset', 'DecathlonDataset', 'CrossValidation'
                'challenge': 'Task01_BrainTumour',  # only need for decathlon:   'Task01_BrainTumour', 'Task02_Heart',
                # 'Task03_Liver0', 'Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas',
                # 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon'
                'seed': 42,  # find the truth in randomness
                'val_frac': 0.2,
                'test_frac': 0.2,
                'use_cache': False,  # if true goes heavy on memory
                'cache_max': sys.maxsize,
                'cache_rate': 0.0,  # 0.0 minimal memory footprint, 1.0 goes heavy on memory
                'num_workers': 8,
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
                'early_stop': {'min_delta': 0.0, 'patience': 0, 'verbose': False, 'mode': 'max'},
            },
        }

    def __str__(self):
        return self.__class__.__name__


if __name__ == '__main__':
    cm = ConfigFile('project_name', 'experiment_name', 'project_base_path', 'dataset_folder_path')
