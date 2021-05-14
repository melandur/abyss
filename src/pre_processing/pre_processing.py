import os
import shutil
import numpy as np
from loguru import logger as log

from src.pre_processing.helpers import ConcatenateImages


class PreProcessing:
    """Whatever your data needs"""

    def __init__(self, params):
        self.params = params
        self.data_path_store = params['tmp']['data_path_store']
        np.random.seed(params['dataset']['seed'])

        self.train_set_cases = None
        self.test_set_cases = None

        self.create_folder_structure()

        self.create_src_dst_dict()
        # self.train_test_split_by_case_names()
        # self.execute_train_test_split()
        # if self.params['dataset']['concatenate_image_files']:
            # self.concatenate_image_files()

    def create_folder_structure(self):
        """Create folder structure to store the data after the pre-processing"""
        for folder in ['imagesTr', 'labelsTr', 'imagesTs', 'labelsTs']:
            folder_path = os.path.join(self.params['project']['dataset_store_path'], folder)
            os.makedirs(folder_path, exist_ok=True)

    def label_conversion(self):
        pass

    def image_conversion(self):
        pass

    def train_test_split_by_case_names(self):
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

    def create_src_dst_dict(self):
        copy_file_manager_dict = {
            'src_path': None,
            'dst_path': None,
        }

        for case_name in self.data_path_store['image'].keys():
            pass

            # print(self.data_path_store[case_name]['image'])
            # print(case_name)
        #     print(case_name)
            # copy_file_manager_dict['src_path']['case'] = self.data_path_store[case_name]['label']
            # copy_file_manager_dict
            # print()
                # copy_helper(self.data_path_store[case_name]['label'], 'labelsTr')
                # for image_tag in self.data_path_store[case_name]['image']:
                #     copy_helper(self.data_path_store[case_name]['image'][image_tag], 'imagesTr')
            #
            # if case_name in self.test_set_cases:
            #     copy_helper(self.data_path_store[case_name]['label'], 'labelsTs')
            #     for image_tag in self.data_path_store[case_name]['image']:
            #         copy_helper(self.data_path_store[case_name]['image'][image_tag], 'imagesTs')




    #
    # def execute_train_test_split(self):
    #     """Copies files to folders: imageTr, labelTr, imageTs, labelTs"""
    #
    #     def copy_helper(src, split_folder):
    #         file_name = os.path.basename(src)
    #         try:
    #             shutil.copy2(src, os.path.join(self.params['project']['dataset_store_path'], split_folder, file_name))
    #         except Exception as e:
    #             log.warning(e)
    #
    #     log.info('Copies data for train test split')
    #     for case_name in self.data_path_store.keys():
    #         if case_name in self.train_set_cases:
    #             copy_helper(self.data_path_store[case_name]['label'], 'labelsTr')
    #             for image_tag in self.data_path_store[case_name]['image']:
    #                 copy_helper(self.data_path_store[case_name]['image'][image_tag], 'imagesTr')
    #
    #         if case_name in self.test_set_cases:
    #             copy_helper(self.data_path_store[case_name]['label'], 'labelsTs')
    #             for image_tag in self.data_path_store[case_name]['image']:
    #                 copy_helper(self.data_path_store[case_name]['image'][image_tag], 'imagesTs')
    #
    # def concatenate_image_files(self):
    #     """Concatenate images together, for example stacking multiple mri modalities"""
    #     for case_name in self.data_path_store.keys():
    #         if case_name in self.train_set_cases:
    #             copy_helper(self.data_path_store[case_name]['label'], 'labelsTr')
    #             for image_tag in self.data_path_store[case_name]['image']:
    #                 copy_helper(self.data_path_store[case_name]['image'][image_tag], 'imagesTr')
    #
    #         if case_name in self.test_set_cases:
    #             copy_helper(self.data_path_store[case_name]['label'], 'labelsTs')
    #             for image_tag in self.data_path_store[case_name]['image']:
    #                 copy_helper(self.data_path_store[case_name]['image'][image_tag], 'imagesTs')
