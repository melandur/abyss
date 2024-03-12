import os

import SimpleITK as sitk
from loguru import logger

from abyss.config import ConfigManager


class PostProcessing(ConfigManager):
    """Post process output"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)

    def __call__(self) -> None:
        if self.params['pipeline_steps']['production']['post_processing']:
            logger.info(f'Run: {self.__class__.__name__}')
            self.process_case_wise()
            self.store_path_memory_file()

    def process_case_wise(self) -> None:
        """Process case wise"""
        for case_name in self.path_memory['inference_paths']:
            logger.info(case_name)
            data = self.load_data(case_name)
            self.apply_largest_connected_component_filter(data, case_name)
            self.store_results(data, case_name)

    def load_data(self, case_name: str) -> sitk.Image:
        """Load data from inference store"""
        data = sitk.ReadImage(self.path_memory['inference_paths'][case_name])
        return data

    @staticmethod
    def apply_largest_connected_component_filter(data: sitk.Image, label: int = 1) -> sitk.Image:
        """Return largest connected component for single label"""
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.SetFullyConnected(True)  # True is less restrictive, gives fewer connected components
        threshold = sitk.BinaryThreshold(data, label, label, label, 0)
        lesions = cc_filter.Execute(threshold)
        rl_filter = sitk.RelabelComponentImageFilter()
        lesions = rl_filter.Execute(lesions)  # sort by size
        filtered_mask = sitk.BinaryThreshold(lesions, label, label, label, 0)
        return filtered_mask

    def store_results(self, data: sitk.Image, case_name: str) -> None:
        """Store processed data"""
        folder_path = os.path.join(self.params['project']['production_store_path'], 'post_processed')
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f'{case_name}.nii.gz')
        sitk.WriteImage(data, file_path)
        self.path_memory['post_processed_paths'][case_name] = file_path
