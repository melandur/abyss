import os

import SimpleITK as sitk

from abyss.config import ConfigManager


class PostProcessing(ConfigManager):
    """Post process output"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)

    def __call__(self):
        self.process_case_wise()

    def process_case_wise(self):
        """Process case wise"""
        for case_name in self.path_memory['inference_store_path']:
            data = self.load_data(case_name)
            self.apply_largest_connected_component_filter(data, case_name)
            self.store_results(data, case_name)

    def load_data(self, case_name):
        """Load data from inference store"""
        data = sitk.ReadImage(self.path_memory['inference_store_path'][case_name])
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

    def store_results(self, data, case_name):
        """Store processed data"""
        file_path = os.path.join(self.params['project']['postprocessed_store_path'], f'{case_name}.nii.gz')
        sitk.WriteImage(data, file_path)
        self.path_memory['postprocessed_dataset_paths'][case_name] = file_path
