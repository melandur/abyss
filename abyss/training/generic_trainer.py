import os
from abc import abstractmethod

import torch
from loguru import logger
from monai.losses import DiceCELoss, DiceLoss
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from abyss.config import ConfigManager
from abyss.training.model import model


class GenericTrainer(ConfigManager):
    """Holds model definitions"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)

        self._val_set = None
        self._test_set = None
        self._train_set = None
        self._device = None
        self._model = model

        self._optimizer = self.__configure_optimizer()
        self._lr_scheduler = self.__configure_scheduler()

        log_path = os.path.join(self.params['project']['result_store_path'], 'log')
        self._log = SummaryWriter(log_dir=log_path, comment='abyss')
        logger.info(f'tensorboard --logdir={log_path}')

        self._early_stop_dict = {
            'patience': self.params['trainer']['early_stop']['patience'],
            'min_delta': self.params['trainer']['early_stop']['min_delta'],
            'counter': 0,
            'val_loss': float('inf'),
        }

        self.__cuda_stuff()

    def __cuda_stuff(self):
        if self.params['trainer']['accelerator'] == 'gpu':
            if not torch.cuda.is_available():
                raise ValueError('GPU not available')
            self._device = torch.device('cuda')
            logger.info(f'Using device: {self._device}')
            self._model = self._model.to(self._device)

        elif self.params['trainer']['accelerator'] == 'cpu':
            self._device = torch.device('cpu')

        else:
            raise ValueError('Invalid trainer.accelerator, choose from [gpu, cpu]')

        if self.params['trainer']['compile']:
            logger.info('Compile model')
            self._model = torch.compile(self._model)

    def _compute_loss(self, output: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Calculate and return loss"""
        loss = self.__criterion(output, ground_truth)
        return loss

    def __criterion(self, output: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:  # TODO: check speed
        """Calculate loss according to criterion"""
        criterion = self.params['training']['criterion']

        if 'mse' == criterion:
            return F.mse_loss(output, ground_truth)

        if 'dice' == criterion:
            dice_loss = DiceLoss()
            return dice_loss(output, ground_truth)  # TODO: Not tested

        if 'cross_entropy' == criterion:
            return F.cross_entropy(output, ground_truth)

        if 'cross_entropy_dice' == criterion:
            dice_ce_loss = DiceCELoss()
            return dice_ce_loss(output, ground_truth)  # TODO: Not tested

        raise ValueError('Invalid criterion settings -> conf.py -> training -> criterion')

    def __configure_scheduler(self) -> torch.optim.lr_scheduler:
        """Returns configured scheduler"""
        max_epochs = self.params['trainer']['max_epochs']
        return LambdaLR(self._optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9)  # TODO: Check this

    def __configure_optimizer(self) -> torch.optim.Optimizer:
        """Returns configured optimizer accordingly to the config file"""
        optimizer_params = self.params['training']['optimizers']

        if optimizer_params['Adam']['active']:
            return torch.optim.Adam(
                params=self._model.parameters(),
                lr=optimizer_params['Adam']['learning_rate'],
                betas=optimizer_params['Adam']['betas'],
                weight_decay=optimizer_params['Adam']['weight_decay'],
                eps=optimizer_params['Adam']['eps'],
                amsgrad=optimizer_params['Adam']['amsgrad'],
            )

        if optimizer_params['SGD']['active']:
            return torch.optim.SGD(
                params=self._model.parameters(),
                lr=optimizer_params['SGD']['learning_rate'],
                momentum=optimizer_params['SGD']['momentum'],
                weight_decay=optimizer_params['Adam']['weight_decay'],
                nesterov=optimizer_params['SGD']['nesterov'],
            )

        raise ValueError('Invalid optimizer settings -> conf.py -> training -> optimizers')

    def _early_stopping(self):
        """Early stop if validation loss is not decreasing anymore"""

    def _load_model(self, path: str) -> None:
        """Load model from path"""
        self._model.load_state_dict(torch.load(path))

    def _save_model(self, path: str) -> None:
        """Save model to path"""
        torch.save(self._model.state_dict(), path)

    @abstractmethod
    def training_step(self, epoch: int, batch: torch.Tensor) -> None:
        """Predict, loss, log, backprop and optimizer step"""

    @abstractmethod
    def validation_step(self, batch: torch.Tensor) -> tuple:
        """Predict, loss, log"""

    @abstractmethod
    def test_step(self, batch: torch.Tensor) -> tuple:
        """Predict, loss, log"""

    @abstractmethod
    def setup(self, stage: str = None) -> torch.utils.data:
        """Initialize datasets"""

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Train dataloader"""

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Validation dataloader"""

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Test dataloader"""
