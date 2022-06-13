import os
from typing import ClassVar

from loguru import logger

from abyss.training.model import Model
from abyss.training.trainer import Trainer


class Training:
    """That's were the gpu is getting sweaty"""

    def __init__(self, config_manager: ClassVar):
        self.config_manager = config_manager
        self.params = config_manager.params
        self.model = Model(self.config_manager)

        if self.params['training']['load_from_checkpoint_path']:
            self.load_from_checkpoint()

    def __call__(self):
        trainer = Trainer(self.config_manager)()
        trainer.fit(self.model)

    def load_from_checkpoint(self):
        """Load checkpoint to proceed training"""
        ckpt_path = self.params['training']['load_from_checkpoint_path']
        if not os.path.isfile(ckpt_path):
            raise FileExistsError(f'Checkpoint file path not found -> {ckpt_path}')
        logger.info(f'Load checkpoint -> {ckpt_path}')
        self.model = self.model.load_from_checkpoint(ckpt_path, config_manager=self.config_manager)
