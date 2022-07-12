import os

import torch
from loguru import logger

from abyss.config import ConfigManager
from abyss.training.model import Model


class Production(ConfigManager):
    """Extract weights from checkpoint"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)
        self.best_checkpoint_path = None

    def __call__(self) -> None:
        self.best_checkpoint_path = self.params['production']['checkpoint_path']
        if self.best_checkpoint_path is None:
            raise ValueError('self.best_checkpoint_path is None -> set path to desired checkpoint')
        if not os.path.isfile(self.best_checkpoint_path):
            raise ValueError('self.best_checkpoint_path is invalid -> check checkpoint path')
        self.extract_model()

    def extract_model(self) -> None:
        """Extract weights from best checkpoint"""
        model = Model(self.params, self.path_memory)
        model_ckpt = model.load_from_checkpoint(
            self.best_checkpoint_path,
            params=self.params,
            path_memory=self.path_memory,
        )
        os.makedirs(self.params['project']['production_store_path'], exist_ok=True)
        file_name = f"{self.params['project']['name']}_{self.params['project']['experiment_name']}"
        export_file_path = os.path.join(self.params['project']['production_store_path'], f'{file_name}.pth')
        torch.save(model_ckpt.state_dict(), export_file_path)
        logger.info(f'Extracted model from checkpoint -> {export_file_path}')
