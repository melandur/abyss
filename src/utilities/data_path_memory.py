from loguru import logger as log

from src.utilities.utils import NestedDefaultDict


class DataPathMemory:
    """Store file paths at different process stages as nested dict"""

    def __init__(self):
        self.path_memory = {
            'structured_dataset_paths': NestedDefaultDict(),
            'preprocessed_dataset_paths': NestedDefaultDict(),
            'train_dataset_paths': NestedDefaultDict(),
            'val_dataset_paths': NestedDefaultDict(),
            'test_dataset_paths': NestedDefaultDict(),
        }

    def show(self):
        """Show data path memory"""
        log.info('Data path memory:\n', self.path_memory)

    def reset(self):
        """Reset path memory"""
        self.path_memory = {
            'structured_dataset_paths': NestedDefaultDict(),
            'preprocessed_dataset_paths': NestedDefaultDict(),
            'train_dataset_paths': NestedDefaultDict(),
            'val_dataset_paths': NestedDefaultDict(),
            'test_dataset_paths': NestedDefaultDict(),
        }
