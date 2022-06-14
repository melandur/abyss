import os

from loguru import logger
from typing_extensions import ClassVar

from abyss.utils import NestedDefaultDict, assure_instance_type


class FileFinder:
    """Creates a nested dictionary, which holds keys:case_names, values: label and image paths"""

    def __init__(self, config_manager: ClassVar):
        params = config_manager.get_params()
        self.dataset_folder_path = params['dataset']['folder_path']
        self.label_search_tags = assure_instance_type(params['dataset']['label_search_tags'], dict)
        self.label_file_type = assure_instance_type(params['dataset']['label_file_type'], list)
        self.data_search_tags = assure_instance_type(params['dataset']['data_search_tags'], dict)
        self.data_file_type = assure_instance_type(params['dataset']['data_file_type'], list)
        self.data_path_store = NestedDefaultDict()

    def __call__(self) -> NestedDefaultDict:
        """Run data analyzer"""
        logger.info(f'Run: {self.__class__.__name__} -> {self.dataset_folder_path}')
        if os.path.isdir(self.dataset_folder_path):
            self.scan_folder()
            self.check_for_missing_files(self.data_search_tags, 'data')
            self.check_for_missing_files(self.label_search_tags, 'label')
            return self.data_path_store
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
            raise AssertionError(f'Case name not found in file name {file_name} and folder name: {root}')
        logger.debug(f'case_name: {case_name} | file_name: {file_name}')
        return case_name

    @staticmethod
    def check_file_search_tag(file_name: str, search_tags: dict) -> bool:
        """True if search tag is in file name"""
        for value in search_tags.values():
            if [x for x in [*value] if x in file_name]:
                return True
        return False

    @staticmethod
    def check_file_type(file_name: str, file_type: dict) -> bool:
        """True if file ends with defined file type"""
        if [x for x in file_type if file_name.endswith(x)]:
            return True
        return False

    def validate_file(self, file_name: str, search_tags: dict, file_type: dict) -> bool:
        """Check if file meets file type and search tag requirement"""
        if self.check_file_search_tag(file_name, search_tags) and self.check_file_type(file_name, file_type):
            return True
        return False

    @staticmethod
    def get_file_search_tag(file_name: str, search_tags: dict) -> str:
        """Returns the found search tag for a certain file name"""
        for key, value in search_tags.items():
            if [x for x in [*value] if x in file_name]:
                return key
        raise ValueError(f'No search tag in file: {file_name} found. Config-file -> check search tags: {search_tags}')

    def scan_folder(self):
        """Walk through the data set folder and assigns file paths to the nested dict"""
        for root, _, files in os.walk(self.dataset_folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    if self.validate_file(file, self.label_search_tags, self.label_file_type):
                        found_tag = self.get_file_search_tag(file, self.label_search_tags)
                        self.data_path_store['label'][self.get_case_name(root, file)][found_tag] = file_path
                    if self.validate_file(file, self.data_search_tags, self.data_file_type):
                        found_tag = self.get_file_search_tag(file, self.data_search_tags)
                        self.data_path_store['data'][self.get_case_name(root, file)][found_tag] = file_path

    def check_for_missing_files(self, search_tag: dict, data_type: str):
        """Check if there are any data/label files are missing"""
        for case_name in self.data_path_store[data_type].keys():
            for tag_name in search_tag.keys():
                if not isinstance(self.data_path_store[data_type][case_name][tag_name], str):
                    raise FileNotFoundError(
                        f'No {tag_name} file found for case {case_name}, check file and '
                        f'search {data_type} tags (case sensitive)'
                    )
