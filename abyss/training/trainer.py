import torch
from loguru import logger
from torch.utils.data import DataLoader

from abyss.training.augmentation.augmentation import transform
from abyss.training.dataset import Dataset
from abyss.training.generic_trainer import GenericTrainer
from abyss.training.helpers.metrics import metric_dice


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
        for epoch in range(self.params['trainer']['max_epochs']):
            logger.info(f'Epoch: {epoch}')
            for batch in self.train_dataloader():
                self.training_step(epoch, batch)
            self._lr_scheduler.step()
            self.validation_iteration(epoch)

    def training_step(self, epoch: int, batch: torch.Tensor) -> None:
        """Predict, loss, log, backprop, optimizer step"""
        data, labels = batch
        data, labels = data.to(self._device), labels.to(self._device)
        self._optimizer.zero_grad()
        output = self._model(data)
        loss = self._compute_loss(output, labels)
        loss.backward()
        self._optimizer.step()
        learning_rate = self._optimizer.defaults['lr']
        self._log.add_scalar('learning_rate', learning_rate, epoch)
        self._log.add_scalar('train_loss', loss.item(), epoch)

    def validation_iteration(self, epoch: int) -> None:
        """Validation iteration"""
        val_dict = {'loss': 0.0, 'dice': 0.0}
        if epoch % self.params['trainer']['check_val_every_n_epoch'] == 0 and epoch != 0:
            for batch in self.val_dataloader():
                val_loss, val_dice = self.validation_step(batch)
                val_dict['loss'] += val_loss
                val_dict['dice'] += val_dice

            val_dict['val_loss'] = val_dict['loss'] / len(self.val_dataloader())
            val_dict['val_dice'] = val_dict['dice'] / len(self.val_dataloader())
            self._log.add_scalar('val_loss', val_dict['val_loss'], epoch)
            self._log.add_scalar('val_dice', val_dict['val_dice'], epoch)

    def validation_step(self, batch: torch.Tensor) -> tuple:
        """Predict, loss, log"""
        data, labels = batch
        data, labels = data.to(self._device), labels.to(self._device)
        output = self._model(data)
        loss = self._compute_loss(output, labels)
        metric_results = metric_dice(output, labels)
        return loss, metric_results

    def test_iteration(self) -> None:
        """Test iteration"""
        test_dict = {'loss': 0.0, 'dice': 0.0}
        for batch in self.test_dataloader():
            test_loss, test_dice = self.test_step(batch)
            test_dict['loss'] += test_loss
            test_dict['dice'] += test_dice

        test_dict['test_loss'] = test_dict['loss'] / len(self.test_dataloader())
        test_dict['test_dice'] = test_dict['dice'] / len(self.test_dataloader())
        self._log.add_scalar('test_loss', test_dict['test_loss'], 0)
        self._log.add_scalar('test_dice', test_dict['test_dice'], 0)

    def test_step(self, batch: torch.Tensor) -> tuple:
        """Predict, loss, log"""
        data, labels = batch
        data, labels = data.to(self._device), labels.to(self._device)
        output = self._model(data)
        loss = self._compute_loss(output, labels)
        metric_results = metric_dice(output, labels)
        return loss, metric_results

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
