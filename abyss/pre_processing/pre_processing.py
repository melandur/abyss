import os

import numpy as np
import torchio as tio
from loguru import logger
from typing_extensions import ClassVar


class PreProcessing:
    """Preprocess data/labels"""

    # TODO: Subject wise preprocessing

    def __init__(self, config_manager: ClassVar):
        self.config_manager = config_manager
        self.params = config_manager.params
        self.path_memory = config_manager.path_memory
        self.structured_dataset_paths = config_manager.get_path_memory('structured_dataset_paths')
        np.random.seed(config_manager.params['meta']['seed'])
        self.data_transformation = None
        self.label_transformation = None

    def __call__(self):
        logger.info(f'Run: {self.__class__.__name__}')
        self.aggregate_data_transformations()
        self.aggregate_label_transformations()
        self.process(self.label_transformation, 'label')
        self.process(self.data_transformation, 'data')
        self.config_manager.store_path_memory_file()

    def aggregate_data_transformations(self):
        """Add data filter"""
        transforms = []
        params = self.params['pre_processing']['data']
        if params['orient_to_ras']['active']:
            transforms.append(tio.ToCanonical())
        if params['resize']['active']:
            transforms.append(
                tio.Resize(target_shape=params['resize']['dim'], image_interpolation=params['resize']['interpolator'])
            )
        if params['resize']['active']:
            transforms.append(tio.ZNormalization())
        if params['resize']['active']:
            transforms.append(tio.RescaleIntensity())
        self.data_transformation = tio.Compose(transforms)

    def aggregate_label_transformations(self):
        """Add label filters"""
        transforms = []
        params = self.params['pre_processing']['label']
        if params['orient_to_ras']['active']:
            transforms.append(tio.ToCanonical())
        if params['resize']['active']:
            transforms.append(
                tio.Resize(target_shape=params['resize']['dim'], image_interpolation=params['resize']['interpolator'])
            )
        self.label_transformation = tio.Compose(transforms)

    def process(self, transformation: tio.Transform, data_type: str):
        """Applies pre-processing task on data"""
        for case_name in self.structured_dataset_paths[data_type]:
            logger.debug(f'{data_type}: {case_name}')
            for file_tag in self.structured_dataset_paths[data_type][case_name]:
                file_path = self.structured_dataset_paths[data_type][case_name][file_tag]
                subject = tio.Subject(data=tio.ScalarImage(file_path))
                subject = transformation(subject)
                self.save_data(subject, case_name, file_tag, folder_tag=data_type)

    def save_data(self, subject: tio.Subject, case_name: str, file_tag: str, folder_tag: str):
        """Save data to preprocessed data folder as nifti file"""
        new_file_dir = os.path.join(self.params['project']['preprocessed_dataset_store_path'], folder_tag)
        os.makedirs(new_file_dir, exist_ok=True)
        new_file_path = os.path.join(new_file_dir, f'{case_name}_{file_tag}.nii.gz')
        subject.data.save(new_file_path)
        self.path_memory['preprocessed_dataset_paths'][folder_tag][case_name][file_tag] = new_file_path
