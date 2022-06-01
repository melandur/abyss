import os

import monai
import numpy as np
from loguru import logger
from typing_extensions import ClassVar


class PreProcessing:
    """Data reading, going from image to numpy space.
    Check out monai.transforms filters
    """

    def __init__(self, config_manager: ClassVar):
        self.config_manager = config_manager
        self.params = config_manager.params
        self.structured_dataset_paths = config_manager.get_path_memory('structured_dataset_paths')
        np.random.seed(config_manager.params['meta']['seed'])
        self.data_reader_label = None
        self.data_reader_image = None

    def __call__(self):
        logger.info(f'Run: {self.__class__.__name__}')
        self.data_reader_image = self.define_data_reader(self.params['pre_processing']['image_data_reader'])
        self.data_reader_label = self.define_data_reader(self.params['pre_processing']['label_data_reader'])
        self.process_images()
        self.process_labels()
        self.config_manager.store_path_memory_file()

    @staticmethod
    def define_data_reader(data_reader_name) -> ClassVar:
        """Returns a monai supported reader class or custom implementation"""
        if data_reader_name == 'ITKReader':
            data_reader = monai.data.ITKReader()
        elif data_reader_name == 'NibabelReader':
            data_reader = monai.data.NibabelReader()
        elif data_reader_name == 'NumpyReader':
            data_reader = monai.data.NumpyReader()
        elif data_reader_name == 'PILReader':
            data_reader = monai.data.PILReader()
        elif data_reader_name == 'CustomReader':
            raise NotImplementedError('Yet to come')
        else:
            raise NotImplementedError(f'Defined data reader {data_reader_name} is not supported')
        return data_reader

    def read_label_data(self, image_path: str) -> tuple:
        """Returns labels as numpy array and meta-data as dict"""
        image_data, meta_data = self.data_reader_image.get_data(self.data_reader_image.read(image_path))
        return image_data, meta_data

    def read_image_data(self, image_path: str) -> tuple:
        """Returns images as numpy array and meta-data as dict"""
        label_data, meta_data = self.data_reader_label.get_data(self.data_reader_label.read(image_path))
        return label_data, meta_data

    def sequential_process_steps_images(self, image):
        """Add image filter"""
        image = self.some_magic_filter_1(image)
        image = self.some_magic_filter_2(image)
        return image

    def sequential_process_steps_labels(self, label):
        """Add segmentation filters"""
        label = self.label_conversion(label)
        return label

    def process_images(self):
        """Applies pre-processing task on images"""
        for case_name in self.structured_dataset_paths['image']:
            logger.debug(f'Image data: {case_name}')
            for image_name in self.structured_dataset_paths['image'][case_name]:
                file_path = self.structured_dataset_paths['image'][case_name][image_name]
                file_name = os.path.basename(file_path)
                image_data, _ = self.read_image_data(file_path)
                processed_image_data = self.sequential_process_steps_images(image_data)
                self.save_data(processed_image_data, case_name, file_name, folder_tag='image')

    def process_labels(self):
        """Applies pre-processing task on labels"""
        for case_name in self.structured_dataset_paths['label']:
            logger.debug(f'Label data: {case_name}')
            file_path = self.structured_dataset_paths['label'][case_name]
            file_name = os.path.basename(file_path)
            label_data, _ = self.read_label_data(file_path)
            processed_label_data = self.sequential_process_steps_labels(label_data)
            self.save_data(processed_label_data, case_name, file_name, folder_tag='label')

    def save_data(self, image_data, case_name, file_name, folder_tag='image'):
        """Save data to preprocessed data folder as nifti or npz file"""
        file_dir = os.path.join(self.params['project']['preprocessed_dataset_store_path'], folder_tag)
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, file_name)
        monai.data.nifti_writer.write_nifti(image_data, file_path)

        if folder_tag == 'image':
            self.config_manager.path_memory['preprocessed_dataset_paths'][folder_tag][case_name] = file_path
        elif folder_tag == 'label':
            self.config_manager.path_memory['preprocessed_dataset_paths'][folder_tag][case_name] = file_path
        else:
            raise ValueError(f'Folder tag: "{folder_tag}" not found')

    def show_processed_data(self):
        """Show the processed data"""

    @staticmethod
    def label_conversion(label):
        """For the sake of ideas"""
        return label

    @staticmethod
    def image_conversion(image):
        """For the sake of ideas"""
        return image

    @staticmethod
    def some_magic_filter_1(image):
        """For the sake of ideas"""
        return image

    @staticmethod
    def some_magic_filter_2(image):
        """For the sake of ideas"""
        return image

    @staticmethod
    def image_resizing(image):
        """For the sake of ideas"""
        return image
