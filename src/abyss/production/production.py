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
        self.get_best_checkpoint_path()
        self.extract_model()

    def get_best_checkpoint_path(self):
        """Returns specific named checkpoint or the one with the _best tag"""
        check_point_store = os.path.join(self.params['project']['result_store_path'], 'checkpoints')
        checkpoint_name = self.params['production']['checkpoint_name']
        if checkpoint_name is None:
            found_checkpoint_name = [x for x in os.listdir(check_point_store) if '_best' in x]
            if len(found_checkpoint_name) != 1:
                raise ValueError(
                    'Multiple or None checkpoints with "_best" tag, set specific name -> '
                    'config_file -> production -> checkpoint_name'
                )
            self.best_checkpoint_path = os.path.join(check_point_store, found_checkpoint_name[0])
        else:
            self.best_checkpoint_path = os.path.join(check_point_store, checkpoint_name)
            if not os.path.isfile(self.best_checkpoint_path):
                raise ValueError(
                    f'Invalid checkpoint path: {self.best_checkpoint_path},'
                    f'-> config_file -> production -> checkpoint_name'
                )
        logger.info(f'Found checkpoint -> {self.best_checkpoint_path}')

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
