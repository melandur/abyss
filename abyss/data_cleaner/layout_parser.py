import os
from loguru import logger

from abyss.utils import NestedDefaultDict


class LayoutParser:
    """Create corresponding layout pattern"""

    def __init__(self, params: dict, data_path_store: dict):
        self.params = params
        self.data_path_store = data_path_store
        self.label_layout = {}
        self.image_layout = {}
        self.decoded_path_store = NestedDefaultDict()

        self.user_definition_check()

    def __call__(self):
        self.decode_folder_layout()

    def user_definition_check(self):
        """Small user input check of the provided folder layout definitions"""
        case_name_definition = self.params['dataset']['get_case_name_from']
        label_definition = self.clean_definition(self.params['dataset']['label_folder_layout'])
        image_definition = self.clean_definition(self.params['dataset']['image_folder_layout'])
        self.checks(case_name_definition, label_definition, image_definition)
        self.label_layout = self._names_to_dict(self.label_layout, label_definition)
        self.image_layout = self._names_to_dict(self.image_layout, image_definition)

    @staticmethod
    def clean_definition(definition: str) -> list:
        """Remove empty spaces and split folder names by ->"""
        if '->' in definition:
            definition = definition.replace(' ', '')
            return definition.split('->')
        raise ValueError('Missing separator, use "->" to separate folder layers')

    @staticmethod
    def check_definition(definition: list):
        """Check if used words are valid"""
        allowed_names = ['case_folder', 'time_step', 'modality_folder', 'image_files', 'dicom_files']
        count_allowed_names = len(allowed_names)
        count_layout = len(definition)
        diff = len(set(allowed_names) - set(definition))
        if count_allowed_names != count_layout + diff:
            raise ValueError(f'Invalid name in config -> folder_layout, options: {allowed_names}')

    @staticmethod
    def check_case_name_definition(case_name, definition: list):
        """Check if used case name definition is in layout definition"""
        if case_name not in definition:
            raise ValueError(f'Invalid name in config -> get_case_name_from, options: {definition} ')

    def checks(self, case_name_definition, label_definition, image_definition):
        self.check_definition(label_definition)
        self.check_definition(image_definition)
        self.check_case_name_definition(case_name_definition, label_definition)
        self.check_case_name_definition(case_name_definition, image_definition)

    @staticmethod
    def _names_to_dict(layout_store, definition):
        """Create empty dict from layout names"""
        for tag in definition:
            layout_store[tag] = None
        return layout_store

    @staticmethod
    def check_case_name(case_name: str):
        """Extracts specific case name from file name"""
        bad_chars = ['#', '<', '>', '$', '%', '!', '&', '*', "'", '"', '{', '}', '/', ':', '@', '+', '.']
        for bad_char in bad_chars:
            if case_name.count(bad_char) != 0:
                raise AssertionError(f'Filename: {case_name} contains bad char: "{bad_char}"')

    def decode_path(self, layout_store: dict, file_path: str):
        """Decodes file path according to the defined data layout"""
        for i, case_name in enumerate(reversed(layout_store)):
            if i == 0:
                layout_store[case_name] = os.path.basename(file_path)
                continue
            file_path = os.path.dirname(file_path)
            found_case_name = os.path.basename(file_path)
            self.check_case_name(found_case_name)
            layout_store[case_name] = found_case_name
        return layout_store

    def decode_folder_layout(self):
        """Recursive decoding of folder definition to found file paths"""
        logger.trace('Decode folder layout:')
        for dataset_type, file in self.data_path_store.items():
            if dataset_type == 'image':
                for _, file_data in file.items():
                    modality = list(file_data.keys())[0]
                    file_path = file_data[modality]
                    layout_store = self.decode_path(self.image_layout, file_path)
                    logger.trace(layout_store)
                    self.decoded_path_store['image'][layout_store['case_folder']][
                        layout_store['modality_folder']
                    ] = layout_store['image_files']
