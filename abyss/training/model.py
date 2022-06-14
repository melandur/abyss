from typing import ClassVar, Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from abyss.training.augmentation import Augmentation
from abyss.training.dataset import Dataset
from abyss.training.nets import resnet_10


class Model(pl.LightningModule):
    """Holds model definitions"""

    def __init__(self, config_manager: ClassVar):
        super().__init__()
        self.config_manager = config_manager
        self.params = config_manager.get_params()
        self.val_set = None
        self.test_set = None
        self.train_set = None
        self.transforms = Augmentation(self.config_manager).compose_transforms()
        self.net = resnet_10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def setup(self, stage: Optional[str] = None) -> torch.utils.data:
        if stage == 'fit' or stage is None:
            self.train_set = Dataset(self.config_manager, 'train', self.transforms)
            self.val_set = Dataset(self.config_manager, 'val')

        if stage == 'test' or stage is None:
            self.test_set = Dataset(self.config_manager, 'test')

    def compute_loss(self, output: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Returns loss"""
        # loss = torch.tensor([0])
        # for criterion in self.params['training']['criterion']:
        #     if 'mse' in criterion:
        loss = F.mse_loss(output, ground_truth)
        #     if 'cross_entropy' in criterion:
        # loss = F.cross_entropy(output, ground_truth)
        return loss

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Predict, compare, log, backprop"""
        data, label = batch
        output = self(data)
        label = torch.tensor([1]).to(torch.float32)
        loss = self.compute_loss(output, label)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Predict, compare, log"""
        data, label = batch
        output = self(data)
        label = torch.tensor([1]).to(torch.float32)
        loss = self.compute_loss(output, label)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs: list) -> None:
        """Validation epoch"""
        # val_loss, num_items = 0, 0
        # for output in outputs:
        #     val_loss += output['val_loss'].sum().item()
        #     num_items += len(output['val_loss'])
        # mean_val_loss = torch.tensor(val_loss / (num_items + 1e-4))
        # self.log('val_loss', mean_val_loss)

    def test_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Predict, compare, log"""
        data, label = batch
        output = self(data)
        label = torch.tensor([1]).to(torch.float32)
        loss = self.compute_loss(output, label)
        self.log('test_loss', loss)
        return loss

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

    def show_train_batch(self):
        """Visualize ce train batch"""
        ori_dataset = Dataset(self.config_manager, 'train')
        aug_dataset = Dataset(self.config_manager, 'train', self.transforms)

        ori_loader = DataLoader(ori_dataset, 1)
        aug_loader = DataLoader(aug_dataset, 1)

        slice_number = 70
        plt.figure(figsize=(15, 10))
        plt.tight_layout()
        while True:
            for (ori_data, _), (aug_data, _) in zip(ori_loader, aug_loader):
                for modality in range(len(ori_data[0])):
                    plt.subplot(2, 4, modality + 1)
                    plt.title('Original')
                    plt.imshow(ori_data[0, modality, slice_number], cmap='gray')
                    plt.subplot(2, 4, modality + 5)
                    plt.title('Augmented')
                    plt.imshow(aug_data[0, modality, slice_number], cmap='gray')
                plt.draw()
                plt.waitforbuttonpress(0)
