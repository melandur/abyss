import os

import torch
from loguru import logger

from abyss.config import ConfigManager
from abyss.training.model import Model
from abyss.training.trainer import Trainer


class Training(ConfigManager):
    """That's were the gpu is getting sweaty"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)
        self.model = None
        self.checkpoint_is_loaded = False

    def __call__(self) -> None:
        if any(self.params['pipeline_steps']['training'].values()):
            logger.info(f'Run: {self.__class__.__name__}')
            self.model = Model(self.params, self.path_memory)

            if self.params['training']['load_from_checkpoint_path']:
                self.load_checkpoint()
                self.checkpoint_is_loaded = True

            if self.params['training']['load_from_weights_path']:
                self.load_weights()
                self.checkpoint_is_loaded = True

            trainer = Trainer(self.model)()

            if self.params['pipeline_steps']['training']['fit']:
                trainer.fit(self.model)
                self.checkpoint_is_loaded = True

            if self.params['pipeline_steps']['training']['test']:
                if not self.checkpoint_is_loaded:
                    raise ValueError(
                        'Missing checkpoint/weights, set loading path -> config_file -> training -> '
                        'load_from_weights_path or load_from_checkpoint_path'
                    )
                trainer.test(self.model)

    def load_checkpoint(self) -> None:
        """Load checkpoint (weights, optimizer state) to proceed training"""
        ckpt_path = self.params['training']['load_from_checkpoint_path']
        if not os.path.isfile(ckpt_path):
            raise FileExistsError(f'Checkpoint file path not found -> {ckpt_path}')
        logger.info(f'Load checkpoint -> {ckpt_path}')
        self.model = self.model.load_from_checkpoint(ckpt_path)

    def load_weights(self) -> None:
        """Load only weights, resets optimizer state"""
        weights_path = self.params['training']['load_from_weights_path']
        if not os.path.isfile(weights_path):
            raise FileExistsError(f'Checkpoint file path not found -> {weights_path}')
        logger.info(f'Load weights -> {weights_path}')
        self.model.load_state_dict(torch.load(weights_path, map_location='cuda:0'), strict=False)  # TODO: map_location
