import os

import torch
from loguru import logger

from abyss.config import ConfigManager
from abyss.training.model import Model


class ExtractWeights(ConfigManager):
    """Extract weights from checkpoint"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)
        self.best_checkpoint_path = None

    def __call__(self) -> None:
        if self.params['pipeline_steps']['production']['extract_weights']:
            logger.info(f'Run: {self.__class__.__name__}')
            self.get_best_checkpoint_path()
            self.extract_model()

    def get_best_checkpoint_path(self) -> None:
        """Returns specific named checkpoint or the one with the _best tag"""
        check_point_store = os.path.join(self.params['project']['result_store_path'], 'checkpoints')
        checkpoint_name = self.params['production']['checkpoint_name']
        if checkpoint_name is None:
            found_checkpoint_name = [x for x in os.listdir(check_point_store) if '_best' in x]
            if len(found_checkpoint_name) == 0:
                raise ValueError('No checkpoint with "_best" tag, maybe you need to train first')
            if len(found_checkpoint_name) > 1:
                raise ValueError(
                    'Multiple checkpoints with "_best" tag. Delete obsolete ones or set specific name -> '
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
        export_folder_path = os.path.join(self.params['project']['production_store_path'], 'weights')
        os.makedirs(export_folder_path, exist_ok=True)
        export_file_path = os.path.join(export_folder_path, f'{file_name}.pth')
        torch.save(model_ckpt.state_dict(), export_file_path)
        logger.info(f'Extracted weights -> {export_file_path}')
