import json
import os
import sys

from loguru import logger

from abyss.config.config_helpers import (
    check_and_create_folder_structure,
    check_image_search_tag_redundancy,
    check_image_search_tag_uniqueness,
)
from abyss.config_file import ConfigFile
from abyss.utils import NestedDefaultDict


class ConfigManager:
    """The pipelines control center, most parameters can be found here"""

    def __init__(self, load_config_file_path: str = None, load_path_memory: bool = False):
        self.path_memory = {
            'structured_dataset_paths': NestedDefaultDict(),
            'preprocessed_dataset_paths': NestedDefaultDict(),
            'train_dataset_paths': NestedDefaultDict(),
            'val_dataset_paths': NestedDefaultDict(),
            'test_dataset_paths': NestedDefaultDict(),
        }

        if load_config_file_path is None:
            self.params = ConfigFile()()
        elif isinstance(load_config_file_path, str) and os.path.isfile(load_config_file_path):
            self.load_config_file(load_config_file_path)
        else:
            raise FileNotFoundError(f'Was not able to load config file from file path: {load_config_file_path}')

        logger.remove()
        logger.add(sys.stderr, level=self.params['logger']['level'])

        if load_path_memory:
            self.load_path_memory_file()

        check_image_search_tag_redundancy(self.params)
        check_image_search_tag_uniqueness(self.params)
        check_and_create_folder_structure(self.params)

        self.store_config_file()

    def store_config_file(self):
        """Export conf params as json to the config store folder"""
        file_path = os.path.join(self.params['project']['config_store_path'], 'config.json')
        with open(file_path, 'w+', encoding='utf-8') as file:
            file.write(json.dumps(self.params, indent=4))
        logger.debug(f'Config file has been stored to {file_path}')

    def load_config_file(self, file_path: str = None):
        """Export path memory as json to the config store folder"""
        if file_path is None:
            file_path = os.path.join(self.params['project']['config_store_path'], 'config.json')
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                self.params = json.load(file)
        else:
            raise FileNotFoundError(f'Config file not found with file path: {file_path}')
        logger.info(f'Config file has been loaded from {file_path}')

    def store_path_memory_file(self):
        """Export path memory as json to the config store folder"""
        file_path = os.path.join(self.params['project']['config_store_path'], 'path_memory.json')
        with open(file_path, 'w+', encoding='utf-8') as file:
            file.write(json.dumps(self.path_memory, indent=4))
        logger.debug(f'Memory path file has been stored to {file_path}')

    def load_path_memory_file(self):
        """Export path memory as json to the config store folder"""
        file_path = os.path.join(self.params['project']['config_store_path'], 'path_memory.json')
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                self.path_memory = json.load(file)
        else:
            raise FileNotFoundError(f'Path memory file not found with file path: {file_path}')

        # python loads the dicts as default dicts, therefore we need to override those with the nested dicts
        for key in self.path_memory.keys():
            if not self.path_memory[key]:  # check if value for certain key is empty
                self.path_memory[key] = NestedDefaultDict()  # override empty dicts with nested dicts
                self.path_memory[key]['image'] = {}
                self.path_memory[key]['label'] = {}
        logger.info(f'Path memory file has been loaded from {file_path}')
        logger.debug(f'Memory path file contains: {json.dumps(self.path_memory, indent=4)}')


if __name__ == '__main__':
    config_manager = ConfigManager()
