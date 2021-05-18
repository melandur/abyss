import os
import shutil
import numpy as np
from loguru import logger as log
import monai.transforms as tf
import monai
from src.pre_processing.pre_processing_helpers import ConcatenateImages


class PreProcessing:
    """Data reading, going from image to numpy space.
    Check out monai.transforms filters, which can be applied in data augmentation during the training instead"""

    def __init__(self, cm):
        self.cm = cm
        self.params = cm.params
        self.structured_dataset_paths = cm.get_path_memory('structured_dataset_paths')
        np.random.seed(cm.params['dataset']['seed'])

        self.data_reader = self.define_data_reader()
        self.process_images(stack_images=True)
        self.process_labels()

    def define_data_reader(self):
        """Returns a monai supported reader class"""
        data_reader = None
        if self.params['dataset']['data_reader'] == 'ImageReader':
            data_reader = monai.data.ImageReader()
        elif self.params['dataset']['data_reader'] == 'ITKReader':
            data_reader = monai.data.ITKReader()
        elif self.params['dataset']['data_reader'] == 'NibabelReader':
            data_reader = monai.data.NibabelReader()
        elif self.params['dataset']['data_reader'] == 'NumpyReader':
            data_reader = monai.data.NumpyReader()
        elif self.params['dataset']['data_reader'] == 'PILReader':
            data_reader = monai.data.PILReader()
        elif self.params['dataset']['data_reader'] == 'WSIReader':
            data_reader = monai.data.WSIReader()
        else:
            log.error(f'Defined data reader "{self.params["dataset"]["data_reader"]}" is not supported'), exit(1)
        return data_reader

    def read_data(self, image_path):
        """Returns images as numpy array and meta data as dict"""
        image_data, meta_data = self.data_reader.get_data(self.data_reader.read(image_path))
        return image_data, meta_data

    def stack_data(self):
        pass


    def sequential_tasks_images(self):
        pass


    def process_images(self, stack_images):
        """"""
        for case_name in self.structured_dataset_paths['image'].keys():
            for image_name in self.structured_dataset_paths['image'][case_name].keys():
                image_data, meta_data = self.read_data(self.structured_dataset_paths['image'][case_name][image_name])




                print(np.shape(image_data))



    def sequential_tasks_labels(self):
        pass

    def process_labels(self):
        pass
            # print(image)

        #     ppd = PreProcessDefinition(case_name)
        #     x = ppd.start(self.structured_dataset_paths['image'])
        #     print(x)
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
