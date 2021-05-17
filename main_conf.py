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

    def __init__(self):
        project_name = 'BratsExp1'
        experiment_name = 'test'
        project_base_path = r'C:\Users\melandur\Downloads\mytest'
        dataset_folder_path = r'C:\Users\melandur\Desktop\test_v2'

        self.params = ConfigFile(project_name, experiment_name, project_base_path, dataset_folder_path).params
        self.path_memory = DataPathMemory().path_memory

        log.remove()  # fresh start
        log.add(sys.stderr, level=self.params['logger']['level'])

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
