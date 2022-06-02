import os

import torchio as tio
import numpy as np
from loguru import logger
from typing_extensions import ClassVar
from typing import Any


class PreProcessing:

    def __init__(self, config_manager: ClassVar):
        self.config_manager = config_manager
        self.params = config_manager.params
        self.structured_dataset_paths = config_manager.get_path_memory('structured_dataset_paths')
        np.random.seed(config_manager.params['meta']['seed'])
        self.image_transformation = None
        self.label_transformation = None

    def __call__(self):
        logger.info(f'Run: {self.__class__.__name__}')
        self.aggregate_image_transformations()
        self.aggregate_label_transformations()
        self.process_images()
        self.process_labels()
        self.config_manager.store_path_memory_file()

    @staticmethod
    def read_label_data(image_path: str) -> Any:
        """Returns labels as numpy array and meta-data as dict"""
        return tio.Subject(label=tio.ScalarImage(image_path))

    @staticmethod
    def read_image_data(image_path: str) -> Any:
        """Returns images as numpy array and meta-data as dict"""
        return tio.Subject(img=tio.ScalarImage(image_path))

    def aggregate_image_transformations(self):
        """Add image filter"""
        transforms = []
        params = self.params['pre_processing']['image']
        if params['canonical']['active']:
            transforms.append(tio.ToCanonical())
        if params['resize']['active']:
            transforms.append(tio.Resize(target_shape=params['resize']['dim'],
                                         image_interpolation=params['resize']['interpolator']))
        if params['resize']['active']:
            transforms.append(tio.ZNormalization())
        if params['resize']['active']:
            transforms.append(tio.RescaleIntensity())
        self.image_transformation =  tio.Compose(transforms)

    def aggregate_label_transformations(self):
        """Add label filters"""
        transforms = []
        params = self.params['pre_processing']['label']
        if params['canonical']['active']:
            transforms.append(tio.ToCanonical())
        if params['resize']['active']:
            transforms.append(tio.Resize(target_shape=params['resize']['dim'],
                                         image_interpolation=params['resize']['interpolator']))
        self.label_transformation = tio.Compose(transforms)

    def process_images(self):
        """Applies pre-processing task on images"""
        for case_name in self.structured_dataset_paths['image']:
            logger.debug(f'Image data: {case_name}')
            for image_name in self.structured_dataset_paths['image'][case_name]:
                file_path = self.structured_dataset_paths['image'][case_name][image_name]
                file_name = os.path.basename(file_path)
                subject = self.read_image_data(file_path)
                subject = self.image_transformation(subject)
                self.save_data(subject, case_name, file_name, folder_tag='image')

    def process_labels(self):
        """Applies pre-processing task on labels"""
        for case_name in self.structured_dataset_paths['label']:
            logger.debug(f'Label data: {case_name}')
            file_path = self.structured_dataset_paths['label'][case_name]
            file_name = os.path.basename(file_path)
            subject = self.read_label_data(file_path)
            subject = self.label_transformation(subject)
            self.save_data(subject, case_name, file_name, folder_tag='label')

    def save_data(self, subject: tio.Subject, case_name: str, file_name: str, folder_tag: str):
        """Save data to preprocessed data folder as nifti or npz file"""
        file_dir = os.path.join(self.params['project']['preprocessed_dataset_store_path'], folder_tag)
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, file_name)

        if folder_tag == 'image':
            subject.img.save(file_path)
            self.config_manager.path_memory['preprocessed_dataset_paths'][folder_tag][case_name] = file_path
        elif folder_tag == 'label':
            subject.label.save(file_path)
            self.config_manager.path_memory['preprocessed_dataset_paths'][folder_tag][case_name] = file_path
        else:
            raise ValueError(f'Folder tag: "{folder_tag}" not found')
