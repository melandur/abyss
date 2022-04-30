import os
import secrets

from loguru import logger
from typing_extensions import ClassVar

from abyss.utils import NestedDefaultDict, assure_instance_type


class DataAnalyzer:
    """Creates a nested dictionary, which holds keys:case_names, values: label and image paths"""

    def __init__(self, config_manager: ClassVar):
        self.dataset_folder_path = config_manager.params['dataset']['folder_path']
        self.label_search_tags = assure_instance_type(config_manager.params['dataset']['label_search_tags'], list)
        self.label_file_type = assure_instance_type(config_manager.params['dataset']['label_file_type'], list)
        self.image_search_tags = assure_instance_type(config_manager.params['dataset']['image_search_tags'], dict)
        self.image_file_type = assure_instance_type(config_manager.params['dataset']['image_file_type'], list)
        self.data_path_store = NestedDefaultDict()

    def __call__(self):
        """Run data analyzer"""
        logger.info(f'Run: {self.__class__.__name__} -> {self.dataset_folder_path}')
        if os.path.isdir(self.dataset_folder_path):
            self.scan_folder()
        else:
            raise NotADirectoryError(str(self.dataset_folder_path))

    def check_file_search_tag_label(self, file_name: str) -> bool:
        """True if label search tag is in file name"""
        if [x for x in self.label_search_tags if x in file_name]:
            return True
        return False

    def check_file_type_label(self, file_name: str) -> bool:
        """True if label file ends with defined file type"""
        if [x for x in self.label_file_type if file_name.endswith(x)]:
            return True
        return False

    def check_file_search_tag_image(self, file_name: str) -> bool:
        """True if image search tag is in file name"""
        for value in self.image_search_tags.values():
            if [x for x in [*value] if x in file_name]:
                return True
        return False

    def check_file_type_image(self, file_name: str) -> bool:
        """True if image file ends with defined file type"""
        if [x for x in self.image_file_type if file_name.endswith(x)]:
            return True
        return False

    def get_file_search_tag_image(self, file_name: str) -> str:
        """Returns the found search tag for a certain file name"""
        for key, value in self.image_search_tags.items():
            if [x for x in [*value] if x in file_name]:
                return key
        raise ValueError(f'No search tag for file: {file_name} found. Check file and search image tags')

    def scan_folder(self):
        """Walk through the data set folder and assigns file paths to the nested dict"""
        for root, _, files in os.walk(self.dataset_folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    pseudo_name = secrets.token_urlsafe(16)
                    if self.check_file_search_tag_label(file) and self.check_file_type_label(file):
                        self.data_path_store['label'][pseudo_name] = file_path
                    if self.check_file_search_tag_image(file) and self.check_file_type_image(file):
                        found_tag = self.get_file_search_tag_image(file)
                        self.data_path_store['image'][pseudo_name][found_tag] = file_path

