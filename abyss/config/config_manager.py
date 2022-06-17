import json
import os
import sys

from loguru import logger

from abyss.config.config_helpers import (
    check_and_create_folder_structure,
    check_search_tag_redundancy,
    check_search_tag_uniqueness,
)
from abyss.config_file import ConfigFile
from abyss.utils import NestedDefaultDict


class ConfigManager:
    """Manages config file and path memory file"""

    _shared_state = {}  # borg pattern is used, shared class state (params and path_memory state) with children

    def __init__(self, load_config_file_path: str = None):
        self.__dict__ = self._shared_state
        self.params = None
        if load_config_file_path is None:
            self.params = ConfigFile()()
        elif isinstance(load_config_file_path, str) and os.path.isfile(load_config_file_path):
            self.__load_config_file(load_config_file_path)
        else:
            raise FileNotFoundError(f'Was not able to load config file from file path: {load_config_file_path}')
        self.path_memory = NestedDefaultDict()

    def __call__(self):
        self.__load_path_memory_file()
        self.__init_logger()
        self.__config_setting_checks()
        self.__store_config_file()

    def __init_logger(self):
        """Init logger definitions"""
        logger.remove()
        logger.add(sys.stderr, level=self.params['logger']['level'])
        log_file_path = os.path.join(self.params['project']['result_store_path'], 'pipeline.log')
        logger.add(log_file_path, mode='w', level='TRACE', format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')

    def __config_setting_checks(self):
        """A few checks for certain problematic config file parts"""
        check_search_tag_redundancy(self.params, 'data')
        check_search_tag_uniqueness(self.params, 'data')
        check_search_tag_redundancy(self.params, 'label')
        check_search_tag_uniqueness(self.params, 'label')
        check_and_create_folder_structure(self.params)

    def __store_config_file(self):
        """Export conf params as json to the config store folder"""
        file_path = os.path.join(self.params['project']['config_store_path'], 'config.json')
        with open(file_path, 'w+', encoding='utf-8') as file:
            file.write(json.dumps(self.params, indent=4))
        logger.debug(f'Config file has been stored to {file_path}')

    def __load_config_file(self, file_path: str = None):
        """Export path memory as json to the config store folder"""
        if file_path is None:
            file_path = os.path.join(self.params['project']['config_store_path'], 'config.json')
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'Config file not found with file path: {file_path}')
        with open(file_path, 'r', encoding='utf-8') as file:
            self.params = json.load(file)
        logger.info(f'Config file has been loaded from {file_path}')

    def __load_path_memory_file(self):
        """Load path memory file from store folder"""
        file_path = os.path.join(self.params['project']['config_store_path'], 'path_memory.json')
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                path_memory = json.load(file)
            logger.info(f'Path memory file has been loaded from {file_path}')
            logger.trace(f'Memory path file contains: {json.dumps(self.path_memory, indent=4)}')
            self.path_memory = NestedDefaultDict(path_memory)

    def store_path_memory_file(self):
        """Export path memory as json to the config store folder"""
        file_path = os.path.join(self.params['project']['config_store_path'], 'path_memory.json')
        with open(file_path, 'w+', encoding='utf-8') as file:
            file.write(json.dumps(self.path_memory, indent=4))
        logger.debug(f'Memory path file has been stored to {file_path}')