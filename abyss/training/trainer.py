import time
import torch
from loguru import logger
from torch.utils.data import DataLoader

from abyss.training.augmentation.augmentation import test_transform, train_transform, val_transform
from abyss.training.dataset import Dataset
from abyss.training.generic_trainer import GenericTrainer


class Trainer(GenericTrainer):
    """Holds model definitions"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)

    def setup(self, stage: str = None) -> torch.utils.data:
        """Define model behaviours"""
        if stage == 'fit':
            logger.info('Setup for training')
            self._train_set = Dataset(self.params, self.path_memory, 'train', train_transform)
            self._val_set = Dataset(self.params, self.path_memory, 'val', val_transform)
            self.training_iteration()

        if stage == 'test':
            logger.info('Setup for testing')
            self._test_set = Dataset(self.params, self.path_memory, 'test', test_transform)
            self.test_iteration()

    def training_iteration(self) -> None:
        """Training iteration"""
        logger.info('Training started')
        for self._epoch in range(self.params['trainer']['total_epochs']):
            time_start = time.time()
            self._model.train()
            for batch in self.train_dataloader():
                self.training_step(batch)

            self._lr_schedulers['lr_rate'].step(self._epoch)
            self.validation_iteration()
            self._check_early_stopping()
            self._check_save_model()

            learning_rate = self._optimizer.param_groups[0]['lr']
            self._log.add_scalar('learning_rate', learning_rate, self._epoch)
            train_loss_average = sum(self._losses['train']) / len(self._losses['train'])
            self._losses['train'] = []  # reset every epoch
            self._log.add_scalar('train_loss', train_loss_average, self._epoch)
            self._log.flush()
            self._execution_time(f'Epoch {self._epoch}', time_start)

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
        if self._epoch > self.params['trainer']['lr_scheduler']['warmup_epochs']:
            if self._epoch % self.params['trainer']['val_epoch'] == 0 and self._epoch != 0:
                self._model.eval()
                with torch.no_grad():
                    for batch in self.val_dataloader():
                        self.validation_step(batch)
                    self._metrics.aggregate('val')
                    val_loss_average = sum(self._losses['val']) / len(self._losses['val'])
                    self._early_stopping['current_loss'] = val_loss_average
                    self._losses['val'] = []  # reset
                    logger.info(f'Metrics: {self._metrics.get("dice", "val")}')
                    self._log.add_scalar('val_loss', val_loss_average, self._epoch)
                    self._log.flush()

    def validation_step(self, batch: torch.Tensor) -> None:
        """Predict, loss, log"""
        data, labels = batch
        data, labels = data.to(self._device), labels.to(self._device)
        output = self._inference(data)
        loss = self._compute_loss(output, labels)
        self._losses['val'].append(loss.item())
        self._metrics.calculate(output, labels, 'val')

    def test_iteration(self) -> None:
        """Test iteration"""
        logger.info('Testing started')
        self._load_model('best')
        self._check_device()
        self._model.eval()
        self.counter = 1
        with torch.no_grad():
            for batch in self.test_dataloader():
                self.test_step(batch)
            self._metrics.aggregate('test')
            logger.info(f'Test metrics: {self._metrics.get("dice", "test")}')

    def test_step(self, batch: torch.Tensor):
        """Predict, loss, log"""
        data, labels = batch
        data, labels = data.to(self._device), labels.to(self._device)
        output = self._inference(data)
        output = torch.sigmoid(output)
        output = (output > 0.5).int()
        x = output.detach().cpu().numpy()
        x = x.squeeze()
        x = x[2, :, :, :]
        import SimpleITK as sitk

        img = sitk.GetImageFromArray(x)
        sitk.WriteImage(img, f'{self.counter}.nii.gz')
        self.counter += 1

        self._metrics.calculate(output, labels, 'test')

    def train_dataloader(self) -> DataLoader:
        """Train dataloader"""
        return DataLoader(
            self._train_set,
            batch_size=self.params['training']['batch_size'],
            num_workers=self.params['meta']['num_workers'],
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader"""
        return DataLoader(
            self._val_set,
            batch_size=2,
            num_workers=self.params['meta']['num_workers'],
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader"""
        return DataLoader(
            self._test_set,
            batch_size=1,
            num_workers=self.params['meta']['num_workers'],
            shuffle=False,
        )
