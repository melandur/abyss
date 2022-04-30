import os
from loguru import logger


class LayoutParser:
    """Create corresponding layout pattern"""

    def __init__(self, params: dict):
        self.params = params
        self.label_layout = None
        self.image_layout = None

    def __call__(self):
        label_layout = self.clean_definition(self.params['dataset']['label_folder_layout'])
        image_layout = self.clean_definition(self.params['dataset']['image_folder_layout'])
        self.label_layout = self.get_definition(label_layout)
        self.image_layout = self.get_definition(image_layout)

    @staticmethod
    def clean_definition(definition: str) -> list:
        """Remove empty spaces and split folder names by ->"""
        if '->' in definition:
            definition = definition.replace(' ', '')
            return definition.split('->')
        raise ValueError('Missing separator, use "->" to separate folder layers')

    @staticmethod
    def get_definition(definition: list) -> tuple:
        """Check if used words are valid"""
        allowed_names = ['case_folder', 'time_step', 'modality_folder', 'image_files', 'dicom_files']
        count_allowed_names = len(allowed_names)
        count_layout = len(definition)
        diff = len(set(allowed_names) - set(definition))
        if count_allowed_names == count_layout + diff:
            return tuple(definition)
        raise ValueError(f'Invalid name in meta -> folder_layout, options: {allowed_names}')

    # @staticmethod
    # def get_case_name(root: str, file_name: str) -> str:
    #     """Extracts specific case name from file name"""
    #     case_name = '_'.join(file_name.split('_')[:-1])
    #     if case_name == '':
    #         case_name = os.path.basename(root)
    #     bad_chars = ['#', '<', '>', '$', '%', '!', '&', '*', "'", '"', '{', '}', '/', ':', '@', '+', '.']
    #     for bad_char in bad_chars:
    #         if case_name.count(bad_char) != 0:
    #             raise AssertionError(f'Filename: {file_name} contains bad char: "{bad_char}"')
    #     if case_name is None:
    #         raise AssertionError(f'Case name not found in file name {file_name} and folder name: {root}')
    #     logger.debug(f'case_name: {case_name} | file_name: {file_name}')
    #     return case_name

    def decode_folder_layout_of_found_files(self):
        """adadasd"""

    # def check_for_missing_files(self):
    #     """Check if there are any image/label files are missing"""
    #     for case_name in self.data_path_store['image'].keys():
    #         for tag_name in self.image_search_tags.keys():
    #             if not isinstance(self.data_path_store['image'][case_name][tag_name], str):
    #                 raise FileNotFoundError(
    #                     f'No {tag_name} file found for case {case_name}, check file and '
    #                     f'search image tags (case sensitive)'
    #                 )
    #
    #         if not isinstance(self.data_path_store['label'][case_name], str):
    #             raise FileNotFoundError(
    #                 f'No seg file found for case {case_name}, check file and label search ' f'tags (case sensitive)'
    #             )
