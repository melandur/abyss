import os
import shutil
import numpy as np
from loguru import logger as log
import monai.transforms as tf

from src.pre_processing.pre_processing_helpers import ConcatenateImages


class PreProcessDefinition:
    """The pre processing steps are here defined"""
    def __init__(self, case_name):
        self.start = tf.Compose(
            [
                ConcatenateImages(keys=['image'], case_name)
            ]
        )


class PreProcessing:
    """Whatever your data needs"""

    def __init__(self, cm):
        self.cm = cm
        self.params = cm.params
        self.structured_dataset_paths = cm.get_path_memory('structured_dataset_paths')
        np.random.seed(cm.params['dataset']['seed'])

        self.start_pre_processing()

        # self.train_test_split_by_case_names()
        # self.execute_train_test_split()
        # if self.params['dataset']['concatenate_image_files']:
            # self.concatenate_image_files()



    # @staticmethod
    # def copy_helper(src, split_folder):
    #     file_name = os.path.basename(src)
    #         try:
    #             shutil.copy2(src, os.path.join(self.params['project']['dataset_store_path'], split_folder, file_name))
    #         except Exception as e:
    #             log.warning(e)

    def start_pre_processing(self):
        for case_name in self.structured_dataset_paths['image'].keys():
            ppd = PreProcessDefinition(case_name)
            x = ppd.start(self.structured_dataset_paths['image'])
            print(x)
        # ca = ConcatenateImages()
        # x = da.test(data_dicts)
        # print(x)

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



    def label_conversion(self):
        """For the sake of ideas"""
        # TODO: Expand on preprocessing filters, try to use monai.transforms were possible

    def image_conversion(self):
        """For the sake of ideas"""
        # TODO: Expand on preprocessing filters, try to use monai.transforms were possible

    def some_magic_filter_1(self):
        """For the sake of ideas"""
        # TODO: Expand on preprocessing filters, try to use monai.transforms were possible

    def some_magic_filter_2(self):
        """For the sake of ideas"""
        # TODO: Expand on preprocessing filters, try to use monai.transforms were possible

    def image_resizing(self):
        """For the sake of ideas"""
        # TODO: Expand on preprocessing filters, try to use monai.transforms were possible

    def splitting_rgb_channels(self):
        """For the sake of ideas"""
        # TODO: Expand on preprocessing filters, try to use monai.transforms were possible
