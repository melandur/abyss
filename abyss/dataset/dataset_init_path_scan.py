import json
import os
import shutil

import numpy as np
from loguru import logger as log

from src.utilities.utils import NestedDefaultDict, assure_instance_type


class DataSetInitPathScan:
    """Creates a nested dictionary, which holds keys:case_names, values: label and image paths"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.params = config_manager.params
        self.path_memory = config_manager.path_memory
        self.dataset_folder_path = config_manager.params['dataset']['folder_path']
        self.label_search_tags = assure_instance_type(config_manager.params['dataset']['label_search_tags'], list)
        self.label_file_type = assure_instance_type(config_manager.params['dataset']['label_file_type'], list)
        self.image_search_tags = assure_instance_type(config_manager.params['dataset']['image_search_tags'], dict)
        self.image_file_type = assure_instance_type(config_manager.params['dataset']['image_file_type'], list)

        self.data_path_store = NestedDefaultDict()
        np.random.seed(config_manager.params['dataset']['seed'])
        log.info(f'Init: {self.__class__.__name__}')

        if self.check_folder_path(self.dataset_folder_path):
            self.scan_folder()
            self.check_for_missing_files()
            self.show_dict_findings()
            self.create_structured_dataset()

    @staticmethod
    def get_case_name(file_name):
        """Extracts specific case name from file name"""
        # TODO: Depends heavily on the naming of your data set
        case_name = '_'.join(file_name.split('_')[:-1])
        bad_chars = ['#', '<', '>', '$', '%', '!', '&', '*', "'", '"', '{', '}', '/', ':', '@', '+', '.']
        for bad_char in bad_chars:
            if case_name.count(bad_char) != 0:
                raise AssertionError(f'Filename: {file_name} contains bad char: "{bad_char}"')
        log.debug(f'case_name: {case_name} | file_name: {file_name}')
        return case_name

    @staticmethod
    def check_folder_path(folder_path):
        """True if string is not empty or None"""
        state = False
        if os.path.isdir(folder_path):
            state = True
        return state

    @log.catch
    def check_file_search_tag_label(self, file_name):
        """True if label search tag is in file name"""
        check_search_tag = False
        if [x for x in self.label_search_tags if x in file_name]:
            check_search_tag = True
        return check_search_tag

    def check_file_type_label(self, file_name):
        """True if label file ends with defined file type"""
        check_file_type = False
        if [x for x in self.label_file_type if file_name.endswith(x)]:
            check_file_type = True
        return check_file_type

    def check_file_search_tag_image(self, file_name):
        """True if image search tag is in file name"""
        check_search_tag = False
        if [x for x in self.image_search_tags.values() if x[0] in file_name]:
            check_search_tag = True
        return check_search_tag

    def check_file_type_image(self, file_name):
        """True if image file ends with defined file type"""
        check_file_type = False
        if [x for x in self.image_file_type if file_name.endswith(x)]:
            check_file_type = True
        return check_file_type

    def get_file_search_tag_image(self, file_name):
        """Returns the found search tag for a certain file name"""
        found_search_tag = [x for x in self.image_search_tags.values() if x[0] in file_name][0]
        return [k for k, v in self.image_search_tags.items() if v == found_search_tag][0]

    def scan_folder(self):
        """Walk through the data set folder and assigns file paths to the nested dict"""
        for root, _, files in os.walk(self.dataset_folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    if self.check_file_search_tag_label(file) and self.check_file_type_label(file):
                        self.data_path_store['label'][self.get_case_name(file)] = file_path
                    if self.check_file_search_tag_image(file) and self.check_file_type_image(file):
                        found_tag = self.get_file_search_tag_image(file)
                        self.data_path_store['image'][self.get_case_name(file)][found_tag] = file_path

    def check_for_missing_files(self):
        """Check if there are any image/label files are missing"""
        for case_name in self.data_path_store['image'].keys():
            for tag_name in self.image_search_tags.keys():
                if not isinstance(self.data_path_store['image'][case_name][tag_name], str):
                    raise AssertionError(f'No {tag_name} file found for {case_name}, check file and search image tags')
            if not isinstance(self.data_path_store['label'][case_name], str):
                raise AssertionError(f'No seg file found for {case_name}, check file and label search tags')

    def create_structured_dataset(self):
        """Copies the found file to an image/label folder for further pre-processing"""

        def copy_helper(src, folder_name, case_name, tag_name):
            """Copy and renames files by their case and tag name, keeps file extension, returns the new file path"""
            # if isinstance(src, str) and os.path.isfile(src):
            file_name = os.path.basename(src)
            file_extension = file_name.split(os.extsep, 1)[1]
            new_file_name = f'{case_name}_{tag_name}.{file_extension}'
            dst_file_path = os.path.join(
                self.params['project']['structured_dataset_store_path'], folder_name, new_file_name
            )
            os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
            shutil.copy2(src=src, dst=dst_file_path)
            return dst_file_path

        # copy files from original dataset to structured dataset and create file path dict
        log.info('Copying original dataset into structured dataset')
        for case_name in self.data_path_store['image'].keys():
            for tag_name in self.image_search_tags.keys():  # copy images
                self.path_memory['structured_dataset_paths']['image'][case_name][tag_name] = copy_helper(
                    src=self.data_path_store['image'][case_name][tag_name],
                    folder_name='image',
                    case_name=case_name,
                    tag_name=tag_name,
                )

            # copy labels
            self.path_memory['structured_dataset_paths']['label'][case_name] = copy_helper(
                src=self.data_path_store['label'][case_name], folder_name='label', case_name=case_name, tag_name='seg'
            )

        self.config_manager.store_path_memory_file()

    @log.catch
    def show_dict_findings(self):
        """Summaries and shows the findings"""
        log.trace(f'Dataset scan found: {json.dumps(self.data_path_store, indent=4)}')

        count_labels = 0
        count_images = {}
        for image_tag in self.image_search_tags.keys():
            count_images[image_tag] = 0

        for case in self.data_path_store['image'].keys():
            for image_tag, image_path in self.data_path_store['image'][case].items():
                if os.path.isfile(image_path):
                    count_images[image_tag] += 1

        for _, label_path in self.data_path_store['label'].items():
            if os.path.isfile(label_path):
                count_labels += 1

        stats_dict = {
            'Total cases': len(self.data_path_store['image'].keys()),
            'Labels': count_labels,
            'Images': count_images,
        }

        log.info(f'Dataset scan overview: {json.dumps(stats_dict, indent=4)}')


if __name__ == '__main__':
    ds = DataSetInitPathScan()
