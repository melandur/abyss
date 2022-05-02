import os
from loguru import logger

from abyss.utils import NestedDefaultDict


class LayoutParser:
    """Create corresponding layout pattern"""

    def __init__(self, params: dict):
        self.params = params
        self.label_layout = {}
        self.image_layout = {}

    def __call__(self):
        label_definition = self.clean_definition(self.params['dataset']['label_folder_layout'])
        image_definition = self.clean_definition(self.params['dataset']['image_folder_layout'])
        self.check(label_definition)
        self.check(image_definition)
        self.label_layout = self._names_to_dict(self.label_layout, label_definition)
        self.image_layout = self._names_to_dict(self.image_layout, image_definition)

        print(self.label_layout)
        for x in reversed(self.label_layout):
            print(x)

    @staticmethod
    def clean_definition(definition: str) -> list:
        """Remove empty spaces and split folder names by ->"""
        if '->' in definition:
            definition = definition.replace(' ', '')
            return definition.split('->')
        raise ValueError('Missing separator, use "->" to separate folder layers')

    @staticmethod
    def check(definition: list):
        """Check if used words are valid"""
        allowed_names = ['case_folder', 'time_step', 'modality_folder', 'image_files', 'dicom_files']
        count_allowed_names = len(allowed_names)
        count_layout = len(definition)
        diff = len(set(allowed_names) - set(definition))
        if count_allowed_names != count_layout + diff:
            raise ValueError(f'Invalid name in meta -> folder_layout, options: {allowed_names}')

    @staticmethod
    def _names_to_dict(layout_store, definiton):
        """Create empty dict from layout names"""
        for tag in definiton:
            layout_store[tag] = None
        return layout_store

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

    def decode_path(self, file_path):
        """Decodes"""
        # for layout_name in reversed(self.image_layout):

    def decode_folder_layout_of_found_files(self, data_path_store):
        """adadasd"""
        final_path_store = NestedDefaultDict()
        for dataset, file in data_path_store.items():
            if dataset == 'image':
                for _, file_data in file.items():
                    modality = list(file_data.keys())[0]
                    file_path = file_data[modality]
                    self.decode_path(file_path)

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
