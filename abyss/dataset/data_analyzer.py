import os

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
            self.check_for_missing_files()
        else:
            raise NotADirectoryError(str(self.dataset_folder_path))

    @staticmethod
    def get_case_name(root: str, file_name: str) -> str:
        """Extracts specific case name from file name"""
        case_name = '_'.join(file_name.split('_')[:-1])
        if case_name == '':
            case_name = os.path.basename(root)
        bad_chars = ['#', '<', '>', '$', '%', '!', '&', '*', "'", '"', '{', '}', '/', ':', '@', '+', '.']
        for bad_char in bad_chars:
            if case_name.count(bad_char) != 0:
                raise AssertionError(f'Filename: {file_name} contains bad char: "{bad_char}"')
        if case_name is None:
            raise AssertionError(f'Case name not found in file and folder name')
        logger.debug(f'case_name: {case_name} | file_name: {file_name}')
        return case_name

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
                    if self.check_file_search_tag_label(file) and self.check_file_type_label(file):
                        self.data_path_store['label'][self.get_case_name(root, file)] = file_path
                    if self.check_file_search_tag_image(file) and self.check_file_type_image(file):
                        found_tag = self.get_file_search_tag_image(file)
                        self.data_path_store['image'][self.get_case_name(root, file)][found_tag] = file_path

    def check_for_missing_files(self):
        """Check if there are any image/label files are missing"""
        for case_name in self.data_path_store['image'].keys():
            for tag_name in self.image_search_tags.keys():
                if not isinstance(self.data_path_store['image'][case_name][tag_name], str):
                    raise FileNotFoundError(f'No {tag_name} file found for case {case_name}, check file and '
                                            f'search image tags (case sensitive)')

            # if not isinstance(self.data_path_store['label'][case_name], str):
            #     raise FileNotFoundError(f'No seg file found for case {case_name}, check file and label search '
            #                             f'tags (case sensitive)')
