import os
import numpy as np
from loguru import logger as log

from src.utilities.utils import NestedDefaultDict, assure_instance_type


class TrainDataSetPathScan:
    """Creates a nested train dictionary, which holds keys:case_names, values: label and image paths"""

    def __init__(self, params):
        self.params = params
        self.dataset_path = params['project']['dataset_store_path']
        self.label_search_tags = assure_instance_type(params['dataset']['label_search_tags'], list)
        self.label_file_type = assure_instance_type(params['dataset']['label_file_type'], list)
        self.image_search_tags = assure_instance_type(params['dataset']['image_search_tags'], dict)
        self.image_file_type = assure_instance_type(params['dataset']['image_file_type'], list)

        np.random.seed(params['dataset']['seed'])
        self.train_data_path_store = NestedDefaultDict()
        self.test_data_path_store = NestedDefaultDict()
        self.val_data_path_store = NestedDefaultDict()

        log.info(f'Init: {self.__class__.__name__}')

        if self.check_folder_path(self.dataset_path):
            self.scan_folder()
            self.create_train_val_split()

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
    def check_if_train_data(self, file_path):
        state = False
        if 'imagesTr' in file_path or 'labelsTr' in file_path:
            state = True
        return state

    @log.catch
    def check_if_test_data(self, file_path):
        state = False
        if 'imagesTs' in file_path or 'labelsTs' in file_path:
            state = True
        return state

    @log.catch
    def scan_folder(self):
        """Walk through the data set folder and assigns file paths to the nested dict"""
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    if self.check_if_train_data(root):
                        data_store = self.train_data_path_store
                    elif self.check_if_test_data(root):
                        data_store = self.test_data_path_store
                    else:
                        data_store = None
                        log.warning('There is a very likely a naming issue in the created train set')

                    if data_store is not None:
                        if self.check_file_search_tag_label(file) and self.check_file_type_label(file):
                            data_store['label'][self.get_case_name(file)] = file_path
                        if self.check_file_search_tag_image(file) and self.check_file_type_image(file):
                            found_tag = self.get_file_search_tag_image(file)
                            data_store['image'][self.get_case_name(file)][found_tag] = file_path

        self.params['tmp']['test_data_path_store'] = self.test_data_path_store

    @log.catch
    def create_train_val_split(self):
        """Split train data into train and val data"""
        count_cases = len(list(self.train_data_path_store['image'].keys()))
        val_set_size = int(self.params['dataset']['val_frac'] * count_cases)
        val_set_cases = list(np.random.choice(list(self.train_data_path_store['image']),
                                              size=val_set_size,
                                              replace=False))
        val_set_cases = [x for x in list(self.train_data_path_store['image'].keys()) if x not in val_set_cases]

        for case in val_set_cases:
            self.val_data_path_store['image'][case] = self.train_data_path_store['image'][case]
            self.train_data_path_store['image'].pop(case)
            self.val_data_path_store['label'][case] = self.train_data_path_store['label'][case]
            self.train_data_path_store['label'].pop(case)

        self.params['tmp']['train_data_path_store'] = self.train_data_path_store
        self.params['tmp']['val_data_path_store'] = self.val_data_path_store

        print(self.train_data_path_store)


if __name__ == '__main__':
    import sys
    from loguru import logger as log
    from main_conf import ConfigManager

    params = ConfigManager(load_conf_file_path=None).params
    log.remove()  # fresh start
    log.add(sys.stderr, level=params['logger']['level'])
    t = TrainDataSetPathScan(params)
    # for x, a in t.train_data_path_store['image'].items():
    #     print(x, a)
