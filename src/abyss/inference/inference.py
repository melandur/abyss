import os

import numpy as np
import SimpleITK as sitk
import torch

from abyss.config import ConfigManager
from abyss.training.nets import nn_unet


class Inference(ConfigManager):
    """Interference, extract weights from the best checkpoint and set load_from_weights_path to file location"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)
        self.model = nn_unet

    def __call__(self) -> None:
        self.load_weights()
        self.model.eval()
        self.predict_case_wise()
        self.store_path_memory_file()

    def load_weights(self):
        """Load weights from defined path"""
        torch.cuda.empty_cache()
        weights_path = self.params['inference']['weights_path']
        if weights_path is None:
            raise AssertionError('Weights file is not defined -> config_file -> training -> load_from_weights_path')
        if not os.path.isfile(weights_path):
            raise ValueError('Weights file path is not valid -> config_file -> training -> load_from_weights_path')
        self.model.load_state_dict(torch.load(weights_path, map_location='cuda:0'), strict=False)  # TODO: map_location

    def predict_case_wise(self):
        """Predict pre processed data"""
        for case_name in self.path_memory['preprocessed_dataset_paths']['data']:
            data = self.concat_data(case_name)
            self.predict(data, case_name)

    def concat_data(self, case_name: str) -> torch.tensor:
        """This needs to be adapted to training/dataset.py -> concat function"""
        img = None
        for idx, file_tag in enumerate(self.path_memory['preprocessed_dataset_paths']['data'][case_name]):
            tmp_img = sitk.ReadImage(self.path_memory['preprocessed_dataset_paths']['data'][case_name][file_tag])
            tmp_img = sitk.GetArrayFromImage(tmp_img)
            tmp_img = np.expand_dims(tmp_img, axis=0)
            if idx == 0:
                img = tmp_img
            else:
                img = np.concatenate((img, tmp_img), axis=0)
        return torch.from_numpy(img)

    def predict(self, data, case_name):
        """Predict and store results"""
        output = self.model(data)
        file_path = os.path.join(self.params['project']['inference_store_path'], f'{case_name}.nii.gz')
        sitk.WriteImage(output, file_path)
        self.path_memory['inference_dataset_paths'][case_name] = file_path
