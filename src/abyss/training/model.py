import typing as t

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from abyss.training.augmentation.augmentation import transforms
from abyss.training.dataset import Dataset
from abyss.training.helpers import apply_criterion, get_optimizer, log_metrics
from abyss.training.nets import nn_unet


class Model(pl.LightningModule):
    """Holds model definitions"""

    def __init__(self, params, path_memory) -> None:
        super().__init__()
        self.params = params
        self.path_memory = path_memory
        self.val_set = None
        self.test_set = None
        self.train_set = None
        self.net = nn_unet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward step"""
        y = self.net(x)  # pylint: disable=invalid-name
        return y

    def setup(self, stage: t.Optional[str] = None) -> torch.utils.data:
        """Define model behaviours"""
        if stage == 'fit' or stage is None:
            self.train_set = Dataset(self.params, self.path_memory, 'train', transforms)
            self.val_set = Dataset(self.params, self.path_memory, 'val')

        if stage == 'test' or stage is None:
            self.test_set = Dataset(self.params, self.path_memory, 'test')

    def compute_loss(self, output: torch.Tensor, ground_truth: torch.Tensor, stage: str) -> torch.Tensor:
        """Calculate and return loss"""
        loss = apply_criterion(self.params, output, ground_truth)
        self.log(f'{stage}_loss', loss.item(), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Predict, loss, log, (backprop and optimizer step done by lightning)"""
        data, label = batch
        output = self(data)
        log_metrics(self, output, label, 'train')
        loss = self.compute_loss(output, label, 'train')
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Predict, loss, log"""
        data, label = batch
        _ = batch_idx
        output = self(data)
        plt.imshow(label.cpu().numpy()[0, 0, :, :, 80])
        plt.show()
        plt.imshow(label.cpu().numpy()[0, 1, :, :, 80])
        plt.show()
        plt.imshow(label.cpu().numpy()[0, 2, :, :, 80])
        plt.show()
        plt.imshow(label.cpu().numpy()[0, 3, :, :, 80])
        plt.show()
        plt.imshow(output.cpu().numpy()[0, 0, :, :, 80])
        plt.show()
        plt.imshow(output.cpu().numpy()[0, 1, :, :, 80])
        plt.show()
        plt.imshow(output.cpu().numpy()[0, 2, :, :, 80])
        plt.show()
        plt.imshow(output.cpu().numpy()[0, 3, :, :, 80])
        plt.show()
        log_metrics(self, output, label, 'val')
        _ = self.compute_loss(output, label, 'val')

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Predict, loss, log"""
        data, label = batch
        _ = batch_idx
        output = self(data)
        log_metrics(self, output, label, 'test')
        _ = self.compute_loss(output, label, 'test')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers"""
        return get_optimizer(self.params, self.parameters)

    def train_dataloader(self) -> DataLoader:
        """Train dataloader"""
        return DataLoader(
            self.train_set,
            batch_size=self.params['training']['batch_size'],
            num_workers=self.params['meta']['num_workers'],
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader"""
        return DataLoader(
            self.val_set,
            batch_size=self.params['training']['batch_size'],
            num_workers=self.params['meta']['num_workers'],
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader"""
        return DataLoader(
            self.test_set,
            batch_size=self.params['training']['batch_size'],
            num_workers=self.params['meta']['num_workers'],
        )
