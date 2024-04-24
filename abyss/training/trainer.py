import torch
from loguru import logger
from torch.utils.data import DataLoader

from abyss.training.augmentation.augmentation import transform
from abyss.training.dataset import Dataset
from abyss.training.generic_trainer import GenericTrainer
from abyss.training.metrics import metric_dice


class Trainer(GenericTrainer):
    """Holds model definitions"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)

    def setup(self, stage: str = None) -> torch.utils.data:
        """Define model behaviours"""
        if stage == 'fit':
            logger.info('Setup model for training')
            self._train_set = Dataset(self.params, self.path_memory, 'train', transform)
            self._val_set = Dataset(self.params, self.path_memory, 'val')
            self.training_iteration()

        if stage == 'test':
            logger.info('Setup model for testing')
            self._test_set = Dataset(self.params, self.path_memory, 'test')
            self.test_iteration()

    def training_iteration(self) -> None:
        """Training iteration"""
        self._model.train()
        for self._epoch in range(self.params['trainer']['max_epochs']):
            logger.info(f'Epoch: {self._epoch}')
            for batch in self.train_dataloader():
                self.training_step(batch)
            self._lr_scheduler.step()
            self.validation_iteration()
            self._check_early_stopping()
            self._check_save_model()

            learning_rate = self._optimizer.param_groups[0]['lr']
            self._log.add_scalar('learning_rate', learning_rate, self._epoch)
            train_loss_average = sum(self._losses['train']) / len(self._losses['train'])
            self._losses['train'] = []  # reset every epoch
            self._log.add_scalar('train_loss', train_loss_average, self._epoch)

    def training_step(self, batch: torch.Tensor) -> None:
        """Predict, loss, log, backprop, optimizer step"""
        data, labels = batch
        data, labels = data.to(self._device), labels.to(self._device)
        self._optimizer.zero_grad()
        output = self._model(data)
        loss = self._compute_loss(output, labels)
        self._losses['train'].append(loss.item())
        loss.backward()
        self._optimizer.step()

    def validation_iteration(self) -> None:
        """Validation iteration"""
        if self._epoch % self.params['trainer']['check_val_every_n_epoch'] == 0 and self._epoch != 0:
            self._model.eval()
            with torch.no_grad():
                for batch in self.val_dataloader():
                    val_loss, val_dice = self.validation_step(batch)
                    self._losses['val'].append(val_loss)
                    self._metrics['val']['dice'].append(val_dice)

                val_loss_average = sum(self._losses['val']) / len(self._losses['val'])
                self._early_stopping['current_loss'] = val_loss_average
                self._losses['val'] = []  # reset
                self._log.add_scalar('val_loss_average', val_loss_average, self._epoch)
                val_dice_average = sum(self._metrics['val']['dice']) / len(self._metrics['val']['dice'])
                self._metrics['val']['dice'] = []
                self._log.add_scalar('val_dice_average', val_dice_average, self._epoch)

    def validation_step(self, batch: torch.Tensor) -> tuple:
        """Predict, loss, log"""
        data, labels = batch
        data, labels = data.to(self._device), labels.to(self._device)
        output = self._model(data)
        loss = self._compute_loss(output, labels)
        metric_results = metric_dice(output, labels)  # todo: needs to hold multiple metrics, returns dict
        return loss.item(), metric_results.item()

    def test_iteration(self) -> None:
        """Test iteration"""
        self._model.eval()
        with torch.no_grad():
            for batch in self.test_dataloader():
                test_loss, test_dice = self.test_step(batch)
                self._losses['test'].append(test_loss)
                self._metrics['test']['dice'].append(test_dice)

        test_loss_average = sum(self._losses['test']) / len(self._losses['test'])
        self._losses['test'] = []
        self._log.add_scalar('test_loss', test_loss_average, 0)
        test_dice_average = sum(self._metrics['test']['dice']) / len(self._metrics['test']['dice'])
        self._log.add_scalar('test_dice', test_dice_average, 0)

    def test_step(self, batch: torch.Tensor) -> tuple:
        """Predict, loss, log"""
        data, labels = batch
        data, labels = data.to(self._device), labels.to(self._device)
        output = self._model(data)
        loss = self._compute_loss(output, labels)
        metric_results = metric_dice(output, labels)
        return loss.item(), metric_results.item()

    def train_dataloader(self) -> DataLoader:
        """Train dataloader"""
        return DataLoader(
            self._train_set,
            batch_size=self.params['training']['batch_size'],
            num_workers=self.params['meta']['num_workers'],
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader"""
        return DataLoader(
            self._val_set,
            batch_size=self.params['training']['batch_size'],
            num_workers=self.params['meta']['num_workers'],
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader"""
        return DataLoader(
            self._test_set,
            batch_size=self.params['training']['batch_size'],
            num_workers=self.params['meta']['num_workers'],
        )
