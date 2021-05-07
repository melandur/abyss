import os
import json
from loguru import logger as log

from src.utils import NestedDefaultDict, assure_instance_type


class DataSetInitPathScan:
    """Creates a nested dictionary, which holds keys:case_names, values: label and image paths"""

    def __init__(self, params):
        self.dataset_path = params['dataset']['folder_path']
        self.label_search_tags = assure_instance_type(params['dataset']['label_search_tags'], list)
        self.label_file_type = assure_instance_type(params['dataset']['label_file_type'], list)
        self.image_search_tags = assure_instance_type(params['dataset']['image_search_tags'], dict)
        self.image_file_type = assure_instance_type(params['dataset']['image_file_type'], list)
        self.data_path_store = NestedDefaultDict()
        log.info(f'Init: {self.__class__.__name__}')

        if self.check_folder_path(self.dataset_path):
            self.scan_folder()

    @staticmethod
    def get_case_name(file_name):
        """Extracts specific case name from file name"""
        # TODO: Depends heavily on the naming of your data set
        case_name = '_'.join(file_name.split('_')[:-1])
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

    @log.catch
    def check_file_type_label(self, file_name):
        """True if label file ends with defined file type"""
        check_file_type = False
        if [x for x in self.label_file_type if file_name.endswith(x)]:
            check_file_type = True
        return check_file_type

    @log.catch
    def check_file_search_tag_image(self, file_name):
        """True if image search tag is in file name"""
        check_search_tag = False
        if [x for x in [*self.image_search_tags.values()] if x[0] in file_name]:
            check_search_tag = True
        return check_search_tag

    @log.catch
    def check_file_type_image(self, file_name):
        """True if image file ends with defined file type"""
        check_file_type = False
        if [x for x in self.image_file_type if file_name.endswith(x)]:
            check_file_type = True
        return check_file_type

    @log.catch
    def get_file_search_tag_image(self, file_name):
        """Returns the found search tag for a certain file name"""
        found_search_tag = [x for x in [*self.image_search_tags.values()] if x[0] in file_name][0]
        return [k for k, v in self.image_search_tags.items() if v == found_search_tag][0]

    @log.catch
    def scan_folder(self):
        """Walk through the data set folder and assigns file paths to the nested dict"""
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    if self.check_file_search_tag_label(file) and self.check_file_type_label(file):
                        self.data_path_store[self.get_case_name(file)]['label'] = file_path
                    if self.check_file_search_tag_image(file) and self.check_file_type_image(file):
                        found_tag = self.get_file_search_tag_image(file)
                        self.data_path_store[self.get_case_name(file)]['image'][found_tag] = file_path
        self.show_dict_findings()

    @log.catch
    def show_dict_findings(self):
        log.trace(f'Dataset scan found: {json.dumps(self.data_path_store, indent=4)}')

        count_labels = 0
        count_images = {}
        for image_tag in self.image_search_tags.keys():
            count_images[image_tag] = 0

        for case in self.data_path_store.keys():
            for label, label_path in self.data_path_store[case].items():
                if 'label' == label:
                    if os.path.isfile(label_path):
                        count_labels += 1

            for image_tag, image_path in self.data_path_store[case]['image'].items():
                if os.path.isfile(image_path):
                    count_images[image_tag] += 1

        stats_dict = {'Total cases': len(self.data_path_store.keys()),
                      'Labels': count_labels,
                      'Images': count_images}

        log.info(f'Dataset scan overview: {json.dumps(stats_dict, indent=4)}')
