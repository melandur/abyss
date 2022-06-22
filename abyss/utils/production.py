import os
import re

import torch
from loguru import logger

from abyss.config import ConfigManager
from abyss.training.model import Model


class Production(ConfigManager):
    """Read and clean original data"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)
        self.best_checkpoint_path = None
        self.file_name = f"{self.params['project']['name']}_{self.params['project']['experiment_name']}"

    def __call__(self) -> None:
        self.get_best_checkpoint()
        self.extract_model()

    def get_best_checkpoint(self) -> None:
        """Get checkpoint with the lowest val_loss"""
        checkpoints_path = os.path.join(self.params['project']['result_store_path'], 'checkpoints')
        checkpoints = os.listdir(checkpoints_path)
        best_val_loss = 1
        for ckpt in checkpoints:
            tmp_ckpt = ckpt.strip('.ckpt')
            tmp_ckpt = re.sub(r'^.*?val_loss=', '', tmp_ckpt)  # strip from left until val_loss number
            try:
                if float(tmp_ckpt) < best_val_loss:
                    best_val_loss = tmp_ckpt
                    self.best_checkpoint_path = os.path.join(checkpoints_path, ckpt)
            except ValueError:
                pass
        logger.info(f'Identified best checkpoint -> {self.best_checkpoint_path}')

    def extract_model(self) -> None:
        """Extract weights from best checkpoint"""
        model = Model(self.params, self.path_memory)
        model_ckpt = model.load_from_checkpoint(
            self.best_checkpoint_path,
            params=self.params,
            path_memory=self.path_memory,
        )
        os.makedirs(self.params['project']['production_store_path'], exist_ok=True)
        export_file_path = os.path.join(self.params['project']['production_store_path'], f'{self.file_name}.pth')
        torch.save(model_ckpt, export_file_path)
        logger.info(f'Extracted model from checkpoint -> {export_file_path}')


if __name__ == '__main__':
    p = Production()
    p()
