from typing import Optional

import pytorch_lightning as pl
import torch
import torchmetrics
from monai.losses import DiceCELoss, DiceLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader

from abyss.training.augmentation.augmentation import transforms
from abyss.training.dataset import Dataset
from abyss.training.nets import resnet_10


class Model(pl.LightningModule):
    """Holds model definitions"""

    def __init__(self, params, path_memory):
        super().__init__()
        self.params = params
        self.path_memory = path_memory
        self.net = None
        self.val_set = None
        self.test_set = None
        self.train_set = None
        self.net = resnet_10

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
        loss = torch.tensor([0], dtype=torch.float32)
        for criterion in self.params['training']['criterion']:
            if 'mse' == criterion:
                loss += F.mse_loss(output, ground_truth)
            if 'dice' == criterion:
                dice_loss = DiceLoss()
                loss += dice_loss(output, ground_truth)  # TODO: Not tested
            if 'cross_entropy' == criterion:
                loss += F.cross_entropy(output, ground_truth)
            if 'cross_entropy_dice' == criterion:
                dice_ce_loss = DiceCELoss()
                loss += dice_ce_loss(output, ground_truth)  # TODO: Not tested
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
        label = torch.tensor([1]).to(torch.int8)
        loss = self.compute_loss(output, label)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        x = torchmetrics.functional.classification.accuracy(label.type(torch.float32), label)
        self.log('val_acc', x, prog_bar=True, on_epoch=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Predict, compare, log"""
        data, label = batch
        output = self(data)
        label = torch.tensor([1]).to(torch.float32)
        loss = self.compute_loss(output, label)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        x = torchmetrics.functional.classification.accuracy(label.type(torch.float32), label)
        self.log('test_acc', x, prog_bar=True, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers"""
        optimizer_params = self.params['training']['optimizers']
        if optimizer_params['Adam']['active']:
            return torch.optim.Adam(
                params=self.parameters(),
                lr=optimizer_params['Adam']['learning_rate'],
                betas=optimizer_params['Adam']['betas'],
                weight_decay=optimizer_params['Adam']['weight_decay'],
                eps=optimizer_params['Adam']['eps'],
                amsgrad=optimizer_params['Adam']['amsgrad'],
            )
        if optimizer_params['SGD']['active']:
            return torch.optim.SGD(
                params=self.parameters(),
                lr=optimizer_params['Adam']['learning_rate'],
                weight_decay=optimizer_params['Adam']['weight_decay'],
            )
        raise ValueError('Invalid optimizer settings -> conf.py -> training -> optimizers -> ')

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
