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
from abyss.utilities.data_path_memory import DataPathMemory
from abyss.utilities.utils import NestedDefaultDict


class ConfigManager:
    """The pipelines control center, most parameters can be found here"""

    def __init__(self, load_config_file_path: str = None):
        self.project_name = 'BratsExp1'
        self.experiment_name = 'test1'
        self.project_base_path = os.path.join(os.path.expanduser("~"), 'Downloads', 'test_abyss')
        self.dataset_folder_path = os.path.join(os.path.expanduser("~"), 'Downloads', 'test_abyss')

        if load_config_file_path is None:
            self.params = ConfigFile(
                self.project_name,
                self.experiment_name,
                self.project_base_path,
                self.dataset_folder_path,
            )()

        elif isinstance(load_config_file_path, str) and os.path.isfile(load_config_file_path):
            self.load_config_file(load_config_file_path)
            self.overwrite_loaded_project_paths()
        else:
            raise FileNotFoundError(f'Was not able to load config file from file path: {load_config_file_path}')

        self.path_memory = DataPathMemory().path_memory

        logger.remove()  # fresh start
        logger.add(sys.stderr, level=self.params['logger']['level'])

        # Some minor dict check
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

    def overwrite_loaded_project_paths(self):
        """Overwrites the loaded config project paths to assure that the config file works inter machine"""
        experiment_path = os.path.join(self.project_base_path, self.project_name, self.experiment_name)
        self.params['project']['structured_dataset_store_path'] = os.path.join(experiment_path, 'structured_dataset')
        self.params['project']['preprocessed_dataset_store_path'] = os.path.join(
            experiment_path, 'pre_processed_dataset'
        )
        self.params['project']['trainset_store_path'] = os.path.join(experiment_path, 'trainset')
        self.params['project']['result_store_path'] = os.path.join(experiment_path, 'results')
        self.params['project']['augmentation_store_path'] = os.path.join(experiment_path, 'aug_plots')
        self.params['project']['config_store_path'] = os.path.join(experiment_path, 'config_data')

    def load_config_file(self, file_path: str = None):
        """Export path memory as json to the config store folder"""
        if file_path is None:
            file_path = os.path.join(self.params['project']['config_store_path'], 'config.json')
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                self.params = json.load(file)
        else:
            raise FileNotFoundError(f'Config file not found with file path: {file_path}')
        logger.debug(f'Config file has been loaded from {file_path}')
        logger.trace(f'Loaded config file contains: {json.dumps(self.params, indent=4)}')

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
        logger.debug(f'Path memory file has been loaded from {file_path}')
        logger.trace(f'Loaded memory path file contains: {json.dumps(self.path_memory, indent=4)}')

    def get_path_memory(self, path_memory_name: str):
        """Returns the temporary path_memory if available, otherwise loads path_memory from path_memory.json"""
        if self.path_memory[path_memory_name]:
            found_path_memory = self.path_memory[path_memory_name]
        else:
            self.load_path_memory_file()
            found_path_memory = self.path_memory[path_memory_name]
        return found_path_memory


if __name__ == '__main__':
    config_manager = ConfigManager()
