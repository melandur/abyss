import os

import numpy as np
import SimpleITK as sitk
import torch
from loguru import logger

from abyss.config import ConfigManager

# from abyss.training.nets import unet


class Inference(ConfigManager):
    """Interference, extract weights from the best checkpoint and set load_from_weights_path to file location"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)
        self.model = None

    def __call__(self) -> None:
        if self.params['pipeline_steps']['production']['inference']:
            logger.info(f'Run: {self.__class__.__name__}')
            torch.cuda.empty_cache()
            self.model = unet
            self.load_weights()
            self.model.eval()
            self.predict_case_wise()
            self.store_path_memory_file()

    def load_weights(self) -> None:
        """Load weights from defined path"""
        weights_name = self.params['production']['weights_name']
        if weights_name is None:
            raise AssertionError('Weights file is not defined -> config_file -> training -> load_from_weights_path')
        weights_path = os.path.join(self.params['project']['production_store_path'], 'weights', weights_name)
        if not os.path.isfile(weights_path):
            raise ValueError('Weights file path is not valid -> config_file -> training -> load_from_weights_path')
        logger.info(f'Load weights -> {weights_path}')
        self.model.load_state_dict(torch.load(weights_path, map_location='cuda:0'), strict=False)  # TODO: map_location

    def predict_case_wise(self) -> None:
        """Predict pre processed data"""
        for case_name in self.path_memory['pre_processed_dataset_paths']['data']:
            logger.info(case_name)
            data = self.prepare_data(case_name)
            data = self.add_channel_dimension(data)
            self.predict(data, case_name)

    @staticmethod
    def add_channel_dimension(data: torch.Tensor) -> torch.Tensor:
        """Extend current data with batch dimension"""
        data = data[None]  # add dimension at the beginning
        return data

    def prepare_data(self, case_name: str) -> torch.tensor:
        """This needs to be adapted to training/dataset.py -> concat function"""
        img = None
        for idx, file_tag in enumerate(self.path_memory['pre_processed_dataset_paths']['data'][case_name]):
            tmp_img = sitk.ReadImage(self.path_memory['pre_processed_dataset_paths']['data'][case_name][file_tag])
            tmp_img = sitk.GetArrayFromImage(tmp_img)
            tmp_img = np.expand_dims(tmp_img, axis=0)
            if idx == 0:
                img = tmp_img
            else:
                img = np.concatenate((img, tmp_img), axis=0)
        return torch.from_numpy(img)

    def predict(self, data: torch.Tensor, case_name: str) -> None:
        """Predict and store results"""
        output = self.model(data)
        output = output.detach().numpy()
        output = sitk.GetImageFromArray(output)
        folder_path = os.path.join(self.params['project']['production_store_path'], 'inference')
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f'{case_name}.nii.gz')
        sitk.WriteImage(output, file_path)
        self.path_memory['inference_paths'][case_name] = file_path
