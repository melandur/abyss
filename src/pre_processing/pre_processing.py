import os
import shutil
import numpy as np

from loguru import logger as log


class PreProcessing:
    """Whatever your data needs"""

    def __init__(self, params, data_path_store):
        self.params = params
        self.data_path_store = data_path_store
        np.random.seed(params['dataset']['seed'])

        self.train_set_cases = None
        self.test_set_cases = None

        self.create_folder_structure()
        self.train_test_split_case_names()
        self.execute_train_test_split()

    def create_folder_structure(self):
        """Create folder structure to store the data after the pre-processing"""
        for folder in ['imageTr', 'labelTr', 'imageTs', 'labelTs']:
            folder_path = os.path.join(self.params['project']['dataset_store_path'], folder)
            os.makedirs(folder_path, exist_ok=True)

    def label_conversion(self):
        pass

    def image_conversion(self):
        pass

    def train_test_split_case_names(self):
        """Creates a list with case names for train and test set each"""
        count_cases = len(list(self.data_path_store.keys()))
        test_set_size = int(self.params['dataset']['test_frac'] * count_cases)
        self.test_set_cases = list(np.random.choice(list(self.data_path_store),
                                                    size=test_set_size,
                                                    replace=False))
        self.train_set_cases = [x for x in list(self.data_path_store.keys()) if x not in self.test_set_cases]
        assert set(self.test_set_cases) != set(self.train_set_cases), log.warning(
            'Contamination in train & test-set split')
        log.info(f'Train set case count: {len(self.train_set_cases)}\n Train set case: {self.train_set_cases}')
        log.info(f'Test set case count: {len(self.test_set_cases)}\n Test set case: {self.test_set_cases}')

    def execute_train_test_split(self):
        """Copies files to folders: imageTr, labelTr, imageTs, labelTs"""

        def copy_helper(src, split_folder):
            file_name = os.path.basename(src)
            shutil.copy2(src, os.path.join(self.params['project']['dataset_store_path'], split_folder, file_name))

        log.info('Copies data for train test split')
        for case_name in self.data_path_store.keys():
            if case_name in self.train_set_cases:
                copy_helper(self.data_path_store[case_name]['label'], 'labelTr')
                for image_tag in self.data_path_store[case_name]['image']:
                    copy_helper(self.data_path_store[case_name]['image'][image_tag], 'imageTr')

            if case_name in self.test_set_cases:
                copy_helper(self.data_path_store[case_name]['label'], 'labelTs')
                for image_tag in self.data_path_store[case_name]['image']:
                    copy_helper(self.data_path_store[case_name]['image'][image_tag], 'imageTs')
