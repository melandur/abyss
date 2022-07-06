from typing import Optional

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.utils.data import DataLoader

from abyss.training.augmentation.augmentation import transforms
from abyss.training.dataset import Dataset
from abyss.training.helpers.model_helpers import (
    apply_criterion,
    get_configured_optimizer,
)
from abyss.training.nets import nn_unet




class Model(pl.LightningModule):
    """Holds model definitions"""

    def __init__(self, params, path_memory) -> None:
        super().__init__()
        self.params = params
        self.path_memory = path_memory
        self.net = None
        self.val_set = None
        self.test_set = None
        self.train_set = None
        self.net = nn_unet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def setup(self, stage: Optional[str] = None) -> torch.utils.data:
        """Define model behaviours"""
        if stage == 'fit' or stage is None:
            self.train_set = Dataset(self.params, self.path_memory, 'train', transforms)
            self.val_set = Dataset(self.params, self.path_memory, 'val')

        if stage == 'test' or stage is None:
            self.test_set = Dataset(self.params, self.path_memory, 'test')

    def compute_loss(self, output: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Returns loss / sum of losses"""
        loss = apply_criterion(self.params, output, ground_truth)
        return loss

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Predict, compare, log, backprop"""
        data, label = batch
        output = self(data)
        label = torch.tensor([1]).to(torch.float32)
        loss = self.compute_loss(output, label)
        self.log('train_loss', loss.item(), prog_bar=True, on_epoch=True)
        x = torchmetrics.functional.classification.accuracy(label.type(torch.float32), label.to(torch.int8))
        self.log('train_accuracy', x, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Predict, compare, log"""
        data, label = batch
        output = self(data)
        loss = self.compute_loss(output, label)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        x = torchmetrics.functional.classification.accuracy(label.type(torch.float32), label)
        self.log('val_acc', x, prog_bar=True, on_epoch=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Predict, compare, log"""
        data, label = batch
        output = self(data)
        loss = self.compute_loss(output, label)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        x = torchmetrics.functional.classification.accuracy(label.type(torch.float32), label)
        self.log('test_acc', x, prog_bar=True, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers"""
        return get_configured_optimizer(self.params, self.parameters)

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
