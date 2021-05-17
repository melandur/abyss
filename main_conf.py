import os
import sys
import json
from loguru import logger as log

from src.utilities.conf_file import ConfigFile
from src.utilities.data_path_memory import DataPathMemory
from src.utilities.conf_helpers import \
    check_and_create_folder_structure, \
    check_image_search_tag_redundancy, \
    check_image_search_tag_uniqueness


class ConfigManager:
    """The pipelines control center, most parameters can be found here"""

    def __init__(self, load_config_file_path=None):
        self.project_name = 'BratsExp1'
        self.experiment_name = 'test1'
        self.project_base_path = r'C:\Users\melandur\Downloads\mytest'
        self.dataset_folder_path = r'C:\Users\melandur\Desktop\test_v2'

        if load_config_file_path is None:
            self.params = ConfigFile(self.project_name,
                                     self.experiment_name,
                                     self.project_base_path,
                                     self.dataset_folder_path).params

        elif isinstance(load_config_file_path, str) and os.path.isfile(load_config_file_path):
            self.load_config_file(load_config_file_path)
            self.overwrite_loaded_project_paths()
        else:
            log.error(f'Was not able to load config file from file path: {load_config_file_path}'), exit(1)

        self.path_memory = DataPathMemory().path_memory

        log.remove()  # fresh start
        log.add(sys.stderr, level=self.params['logger']['level'])

        # Some minor dict check
        check_image_search_tag_redundancy(self.params)
        check_image_search_tag_uniqueness(self.params)
        check_and_create_folder_structure(self.params)
        self.store_config_file()

    @log.catch
    def store_config_file(self):
        """Export conf params as json to the config store folder"""
        file_path = os.path.join(self.params['project']['config_store_path'], f'config.json')
        with open(file_path, 'w+') as f:
            f.write(json.dumps(self.params, indent=4))
        log.debug(f'Config file has been stored to {file_path}')

    @log.catch
    def overwrite_loaded_project_paths(self):
        """Overwrites the loaded config project paths to assure that the config file works inter machine"""
        experiment_path = os.path.join(self.project_base_path, self.project_name, self.experiment_name)
        self.params['project']['structured_dataset_store_path'] = os.path.join(experiment_path, 'structured_dataset')
        self.params['project']['preprocessed_dataset_store_path'] = os.path.join(experiment_path, 'pre_processed_dataset')
        self.params['project']['trainset_store_path'] = os.path.join(experiment_path, 'trainset')
        self.params['project']['result_store_path'] = os.path.join(experiment_path, 'results')
        self.params['project']['augmentation_store_path'] = os.path.join(experiment_path, 'aug_plots')
        self.params['project']['config_store_path'] = os.path.join(experiment_path, 'config_data')

    @log.catch
    def load_config_file(self, file_path=None):
        """Export path memory as json to the config store folder"""
        if file_path is None:
            file_path = os.path.join(self.params['project']['config_store_path'], f'config.json')
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                self.params = json.load(f)
        else:
            log.error(f'Config file not found with file path: {file_path}')
            exit(1)
        log.trace(f'Loaded config file contains: {json.dumps(self.params, indent=4)}')
        log.debug(f'Config file has been loaded from {file_path}')

    @log.catch
    def store_path_memory_file(self):
        """Export path memory as json to the config store folder"""
        file_path = os.path.join(self.params['project']['config_store_path'], f'path_memory.json')
        with open(file_path, 'w+') as f:
            f.write(json.dumps(self.path_memory, indent=4))
        log.debug(f'Memory path file has been stored to {file_path}')

    @log.catch
    def load_path_memory_file(self):
        """Export path memory as json to the config store folder"""
        file_path = os.path.join(self.params['project']['config_store_path'], f'path_memory.json')
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                self.path_memory = json.load(f)
        else:
            log.error(f'Path memory file not found with file path: {file_path}')
            exit(1)
        log.trace(f'Loaded memory path file contains: {json.dumps(self.path_memory, indent=4)}')
        log.debug(f'Path memory file has been loaded from {file_path}')


if __name__ == '__main__':
    cm = ConfigManager()
