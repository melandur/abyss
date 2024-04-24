import math
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
    """Trainer boilerplate"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)

        self._epoch = None

        self._val_set = None
        self._test_set = None
        self._train_set = None
        self._device = None

        self._metrics = {'train': {'dice': []}, 'val': {'dice': []}, 'test': {'dice': []}}
        self._losses = {'train': [], 'val': [], 'test': []}
        self._early_stopping = {'counter': 0, 'best_loss': float('inf'), 'current_loss': float('inf')}

        self._model = model
        self._optimizer = self.__configure_optimizer()
        self._lr_scheduler = self.__configure_scheduler()

        log_path = os.path.join(self.params['project']['result_store_path'], 'log')
        self._log = SummaryWriter(log_dir=log_path)
        logger.info(f'tensorboard --logdir={log_path}')

        self.__check_device()
        self.__compile_model_option()

    def __check_device(self):
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

    def __compile_model_option(self) -> None:
        """Compile model"""
        if self.params['trainer']['compile']:
            logger.info('Compile model')
            self._model = torch.compile(self._model)

    def _compute_loss(self, output: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Calculate and return loss"""
        loss = self.__criterion(output, ground_truth)
        return loss

    def _check_early_stopping(self) -> bool:
        """Early stop if validation loss is not improving and learning rate is below threshold"""
        patience = self.params['trainer']['early_stop']['patience']
        min_delta = self.params['trainer']['early_stop']['min_delta']
        best_loss = self._early_stopping['best_loss']
        current_loss = self._early_stopping['current_loss']
        min_learning_rate = self.params['trainer']['early_stop']['min_learning_rate']

        if best_loss > current_loss + min_delta:
            self._early_stopping['best_loss'] = current_loss
            self._early_stopping['counter'] = 0
        else:
            self._early_stopping['counter'] += 1

        if self._early_stopping['counter'] > patience:
            if self._optimizer.param_groups[0]['lr'] < min_learning_rate:
                logger.info('Early stopping kicked in')
                self._save_model('final')
                return True

            logger.info('Minimum learning rate not reached, continue training')
            # todo: reduce learning rate or shorter validation check interval
        return False

    def _save_model(self, tag: str = None) -> None:
        """Save model to path"""
        if tag is None:
            tag = str(self._epoch)
        model_name = f'{self.params["project"]["experiment_name"]}_{tag}.pth'
        folder_path = os.path.join(self.params['project']['result_store_path'], 'models')
        os.makedirs(folder_path, exist_ok=True)
        model_path = os.path.join(folder_path, model_name)
        state_dict = {
            'epoch': self._epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._lr_scheduler.state_dict(),
        }
        logger.info(f'Save model -> {model_path}')
        torch.save(state_dict, model_path)

    def _load_model(self, path: str) -> None:
        """Load model from path"""
        checkpoint = torch.load(path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self._epoch = checkpoint['epoch']

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

    def __schedule_plan(self, step, warmup_steps, max_steps, lr_start, lr_end):
        if step < warmup_steps:  # cosine annealing warmup
            return lr_start + (lr_end - lr_start) * (1 - math.cos(step / warmup_steps * math.pi)) / 2

        lr_current = self._optimizer.param_groups[0]['lr']
        return lr_current * (1 - step / max_steps) ** 0.9  # nnunet poly learning rate

    def __configure_scheduler(self) -> torch.optim.lr_scheduler:
        """Returns configured scheduler"""
        max_steps = self.params['trainer']['max_epochs']  # epochs are called steps in lr_scheduler
        warmup_steps = self.params['trainer']['lr_scheduler']['warmup_steps']
        lr_end = self._optimizer.defaults['lr']
        lr_scheduler = LambdaLR(
            self._optimizer,
            lr_lambda=lambda step: self.__schedule_plan(step, warmup_steps, max_steps, 0.0, lr_end),
        )
        return lr_scheduler

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

    @abstractmethod
    def setup(self, stage: str = None) -> torch.utils.data:
        """Initialize datasets"""

    @abstractmethod
    def training_step(self, batch: torch.Tensor) -> None:
        """Predict, loss, log, backprop and optimizer step"""

    @abstractmethod
    def validation_step(self, batch: torch.Tensor) -> tuple:
        """Predict, loss, log"""

    @abstractmethod
    def test_step(self, batch: torch.Tensor) -> tuple:
        """Predict, loss, log"""

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Train dataloader"""

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Validation dataloader"""

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Test dataloader"""
