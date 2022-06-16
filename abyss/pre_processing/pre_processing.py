import os

import numpy as np
import torchio as tio
from loguru import logger

from abyss.config import ConfigManager
from abyss.utils import NestedDefaultDict


class PreProcessing(ConfigManager):
    """Preprocess data/labels"""

    # TODO: Subject wise preprocessing

    def __init__(self, **kwargs):
        super().__init__()
        self._shared_state.update(kwargs)
        self.data_transformation = None
        self.label_transformation = None
        np.random.seed(self.params['meta']['seed'])

    def __call__(self):
        logger.info(f'Run: {self.__class__.__name__}')
        self.path_memory['preprocessed_dataset_paths'] = NestedDefaultDict()
        self.aggregate_data_transformations()
        self.aggregate_label_transformations()
        self.process(self.label_transformation, 'label')
        self.process(self.data_transformation, 'data')
        self.store_path_memory_file()

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
        if params['z_score']['active']:
            transforms.append(tio.ZNormalization())
        if params['rescale_intensity']['active']:
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
        preprocessed_dataset_store_path = self.params['project']['preprocessed_dataset_store_path']
        structured_dataset_paths = self.path_memory['structured_dataset_paths']
        if len(structured_dataset_paths['data']) == 0:
            raise ValueError('Path memory file is empty for structured data, check config_file -> pipeline_steps')
        for case_name in structured_dataset_paths[data_type]:
            logger.debug(f'{data_type}: {case_name}')
            for file_tag in structured_dataset_paths[data_type][case_name]:
                file_path = structured_dataset_paths[data_type][case_name][file_tag]
                subject = tio.Subject(data=tio.ScalarImage(file_path))
                subject = transformation(subject)
                self.save_data(subject, preprocessed_dataset_store_path, case_name, file_tag, data_type)

    def save_data(self, subject: tio.Subject, preproc_store_path: str, case_name: str, file_tag: str, data_type: str):
        """Save data to preprocessed data folder as nifti file"""
        new_file_dir = os.path.join(preproc_store_path, data_type)
        os.makedirs(new_file_dir, exist_ok=True)
        new_file_path = os.path.join(new_file_dir, f'{case_name}_{file_tag}.nii.gz')
        self.path_memory['preprocessed_dataset_paths'][data_type][case_name][file_tag] = new_file_path
        subject.data.save(new_file_path)
