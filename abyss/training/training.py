import os

from loguru import logger

from abyss.config import ConfigManager
from abyss.training.model import Model
from abyss.training.trainer import Trainer


class Training(ConfigManager):
    """That's were the gpu is getting sweaty"""

    def __init__(self, **kwargs):
        super().__init__()
        self._shared_state.update(kwargs)
        self.model = None

    def __call__(self):
        self.model = Model(self.params, self.path_memory)
        if self.params['training']['load_from_checkpoint_path']:
            self.load_from_checkpoint()

        if self.params['training']['dev_show_train_batch']:
            self.model.show_train_batch()

        trainer = Trainer()()
        trainer.fit(self.model)
        trainer.test(self.model)

    def load_from_checkpoint(self):
        """Load checkpoint to proceed training"""
        ckpt_path = self.params['training']['load_from_checkpoint_path']
        if not os.path.isfile(ckpt_path):
            raise FileExistsError(f'Checkpoint file path not found -> {ckpt_path}')
        logger.info(f'Load checkpoint -> {ckpt_path}')
        self.model = self.model.load_from_checkpoint(ckpt_path)
