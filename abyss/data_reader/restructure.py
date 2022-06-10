import os
import shutil
from typing import ClassVar

from loguru import logger

from abyss.utils import assure_instance_type


class Restructure:
    """Restructure original data -> to data/label folder"""

    def __init__(self, config_manager: ClassVar, data_path_store: dict):
        self.config_manager = config_manager
        self.params = config_manager.params
        self.path_memory = config_manager.path_memory
        self.label_search_tags = assure_instance_type(self.params['dataset']['label_search_tags'], dict)
        self.data_search_tags = assure_instance_type(self.params['dataset']['data_search_tags'], dict)
        self.data_path_store = data_path_store

    def __call__(self):
        logger.info(f'Run: {self.__class__.__name__}')
        self.create_structured_dataset(self.label_search_tags, 'label')
        self.create_structured_dataset(self.data_search_tags, 'data')

    def create_structured_dataset(self, search_tags: list, data_type: str):
        """Copy files from original dataset to structured dataset and create file path dict"""
        logger.info(f'Copying original {data_type} to new structure -> 2_pre_processed_dataset')
        for case_name in sorted(self.data_path_store[data_type]):
            for tag_name in search_tags:
                self.path_memory['structured_dataset_paths'][data_type][case_name][tag_name] = self.copy_helper(
                    src=self.data_path_store[data_type][case_name][tag_name],
                    folder_name=data_type,
                    case_name=case_name,
                    tag_name=tag_name,
                )
        self.config_manager.store_path_memory_file()

    def copy_helper(self, src: str, folder_name: str, case_name: str, tag_name: str) -> str:
        """Copy and renames files by their case and tag name, keeps file extension, returns the new file path"""
        if isinstance(src, str) and os.path.isfile(src):
            file_name = os.path.basename(src)
            file_extension = file_name.split(os.extsep, 1)[1]  # split on first . and uses the rest as extension
            new_file_name = f'{case_name}_{tag_name}.{file_extension}'
            dst_file_path = os.path.join(
                self.params['project']['structured_dataset_store_path'], folder_name, new_file_name
            )
            os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
            shutil.copy2(src=src, dst=dst_file_path)
            return dst_file_path
        raise AssertionError(f'Files seems not to exist, case: {case_name}, tag: {tag_name}')
