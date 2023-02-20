from typing import Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from abyss.training.augmentation.augmentation import transforms
from abyss.training.dataset import Dataset
from abyss.training.helpers.log_metrics import log_dice
from abyss.training.helpers.model_helpers import apply_criterion, get_optimizer
from abyss.training.nets import unet


class Model(pl.LightningModule):
    """Holds model definitions"""

    def __init__(self, params, path_memory) -> None:
        super().__init__()
        self.params = params
        self.path_memory = path_memory
        self.val_set = None
        self.test_set = None
        self.train_set = None
        self.net = unet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward step"""
        x = self.net(x)
        return x

    def setup(self, stage: Optional[str] = None) -> torch.utils.data:
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
        # plt.subplots(1, 3)
        # plt.subplot(1, 3, 1)
        # plt.imshow(data.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        # plt.title('data')
        # plt.subplot(1, 3, 2)
        # plt.imshow(label.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        # plt.title('label')
        # plt.subplot(1, 3, 3)
        # plt.imshow(output.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        # plt.title('output')
        # plt.show()
        loss = self.compute_loss(output, label, 'train')
        log_dice(self, output, label, 'train')
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Predict, loss, log"""
        data, label = batch
        output = self(data)
        _ = self.compute_loss(output, label, 'val')
        log_dice(self, output, label, 'val')

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Predict, loss, log"""
        data, label = batch
        output = self(data)
        _ = self.compute_loss(output, label, 'test')
        log_dice(self, output, label, 'test')

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
