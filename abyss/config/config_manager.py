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

    def __init__(self, load_config_file_path: str = None):
        self._params = None
        if load_config_file_path is None:
            self._params = ConfigFile()()
        elif isinstance(load_config_file_path, str) and os.path.isfile(load_config_file_path):
            self.load_config_file(load_config_file_path)
        else:
            raise FileNotFoundError(f'Was not able to load config file from file path: {load_config_file_path}')

        self._path_memory = NestedDefaultDict()
        self._load_path_memory_file()
        self._init_logger()
        self._config_setting_checks()
        self.store_config_file()

    def _init_logger(self):
        """Init logger definitions"""
        logger.remove()
        logger.add(sys.stderr, level=self._params['logger']['level'])
        log_file_path = os.path.join(self._params['project']['result_store_path'], 'pipeline.log')
        logger.add(log_file_path, mode='w', level='TRACE', format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')

    def _config_setting_checks(self):
        """A few checks for certain problematic config file parts"""
        check_search_tag_redundancy(self._params, 'data')
        check_search_tag_uniqueness(self._params, 'data')
        check_search_tag_redundancy(self._params, 'label')
        check_search_tag_uniqueness(self._params, 'label')
        check_and_create_folder_structure(self._params)

    def store_config_file(self):
        """Export conf params as json to the config store folder"""
        file_path = os.path.join(self._params['project']['config_store_path'], 'config.json')
        with open(file_path, 'w+', encoding='utf-8') as file:
            file.write(json.dumps(self._params, indent=4))
        logger.debug(f'Config file has been stored to {file_path}')

    def load_config_file(self, file_path: str = None):
        """Export path memory as json to the config store folder"""
        if file_path is None:
            file_path = os.path.join(self._params['project']['config_store_path'], 'config.json')
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'Config file not found with file path: {file_path}')
        with open(file_path, 'r', encoding='utf-8') as file:
            self._params = json.load(file)
        logger.info(f'Config file has been loaded from {file_path}')

    def get_params(self):
        """Return pipeline params"""
        return self._params

    def store_path_memory_file(self):
        """Export path memory as json to the config store folder"""
        file_path = os.path.join(self._params['project']['config_store_path'], 'path_memory.json')
        with open(file_path, 'w+', encoding='utf-8') as file:
            file.write(json.dumps(self._path_memory, indent=4))
        logger.debug(f'Memory path file has been stored to {file_path}')

    def get_path_memory(self) -> NestedDefaultDict:
        """Returns path memory"""
        return self._path_memory

    def set_path_memory(self, path_memory: NestedDefaultDict):
        """Set path memory"""
        self._path_memory = path_memory
        self.store_path_memory_file()

    def _load_path_memory_file(self):
        """Load path memory file from store folder"""
        file_path = os.path.join(self._params['project']['config_store_path'], 'path_memory.json')
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                path_memory = json.load(file)
            logger.info(f'Path memory file has been loaded from {file_path}')
            logger.trace(f'Memory path file contains: {json.dumps(self._path_memory, indent=4)}')
            self._path_memory = NestedDefaultDict(path_memory)
