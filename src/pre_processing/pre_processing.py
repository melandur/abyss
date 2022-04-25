import os

import monai
import numpy as np
from loguru import logger as log


class PreProcessing:
    """Data reading, going from image to numpy space.
    Check out monai.transforms filters, which can be applied in data augmentation during the training instead
    [B],C,H,W,[D]
    """

    def __init__(self, cm):
        self.cm = cm
        self.params = cm.params
        self.structured_dataset_paths = cm.get_path_memory('structured_dataset_paths')
        np.random.seed(cm.params['dataset']['seed'])

        log.info(f'Init {self.__class__.__name__}')
        self.data_reader = self.define_data_reader()
        self.process_images(stack_images=True)
        self.process_labels()
        self.cm.store_path_memory_file()

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

    def sequential_process_steps_images(self, image):
        # TODO: ADD filters
        return image

    def sequential_process_steps_labels(self, image):
        # TODO: ADD filters
        return image

    def process_images(self, stack_images=False):
        """Applies pre processing task on images"""
        for case_name in self.structured_dataset_paths['image'].keys():
            log.debug(f'Image data: {case_name}')
            tmp_image_store = {}
            for image_name in self.structured_dataset_paths['image'][case_name].keys():
                image_data, meta_data = self.read_data(self.structured_dataset_paths['image'][case_name][image_name])
                processed_image_data = self.sequential_process_steps_images(image_data)
                tmp_image_store[image_name] = processed_image_data
            if stack_images:
                processed_image_data = self.stack_data(tmp_image_store)
            self.save_data(processed_image_data, case_name, folder_tag='image', export_tag='.nii.gz')

    def process_labels(self):
        """Applies pre processing task on labels"""
        for case_name in self.structured_dataset_paths['label'].keys():
            log.debug(f'Label data: {case_name}')
            image_data, meta_data = self.read_data(self.structured_dataset_paths['label'][case_name])
            processed_image_data = self.sequential_process_steps_labels(image_data)
            self.save_data(processed_image_data, case_name, folder_tag='label', export_tag='.nii.gz')

    def save_data(self, image_data, case_name, folder_tag='image', export_tag='.nii.gz'):
        """Save data to preprocessed data folder as nifti or npz file"""
        file_dir = os.path.join(self.params['project']['preprocessed_dataset_store_path'], f'{folder_tag}')
        os.makedirs(file_dir, exist_ok=True)
        if export_tag == '.nii.gz':
            file_path = os.path.join(file_dir, f'{case_name}.nii.gz')
            monai.data.nifti_writer.write_nifti(image_data, file_path)
        elif export_tag == '.npz':
            file_path = os.path.join(file_dir, f'{case_name}.npy')
            with open(file_path, 'wb') as f:
                np.save(f, image_data)
        else:
            log.error(f'Export_tag: "{export_tag}" not found'), exit(1)

        if folder_tag == 'image':
            self.cm.path_memory['preprocessed_dataset_paths'][folder_tag][case_name] = file_path
        elif folder_tag == 'label':
            self.cm.path_memory['preprocessed_dataset_paths'][folder_tag][case_name] = file_path
        else:
            log.error(f'Folder tag: "{folder_tag}" not found'), exit(1)

    @staticmethod
    def stack_data(tmp_image_store):
        """"""
        log.debug('Stacking images')
        tmp_stack = None
        for index, image_name in enumerate(tmp_image_store.keys()):
            if index == 0:
                tmp_stack = np.expand_dims(tmp_image_store[image_name], axis=0)
                log.debug(f'{image_name:<7} index:{index:<4} current dim: {np.shape(tmp_stack)}')
            else:
                tmp_stack = np.concatenate((tmp_stack, np.expand_dims(tmp_image_store[image_name], axis=0)), axis=0)
                log.debug(f'{image_name:<7} index:{index:<4} current dim: {np.shape(tmp_stack)}')
        return tmp_stack

    def show_processed_data(self):
        """"""
        pass

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
