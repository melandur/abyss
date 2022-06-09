import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from abyss.training.augmentation import Augmentation
from abyss.training.dataset import Dataset
from abyss.training.nets import unet


class Model(pl.LightningModule):
    """TODO: Need some"""

    def __init__(self, _config_manager):
        super().__init__()
        self.config_manager = _config_manager
        self.params = _config_manager.params
        self.val_set = None
        self.test_set = None
        self.train_set = None
        self.net = unet

    def forward(self, x):
        return self.net(x)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            augmentation = Augmentation(self.config_manager)
            self.train_set = Dataset(self.config_manager, 'train', augmentation.train_transforms)
            self.val_set = Dataset(self.config_manager, 'val')

        if stage == 'test' or stage is None:
            self.test_set = Dataset(self.config_manager, 'test')

    def compute_loss(self, output, ground_truth):
        """Returns loss"""
        loss = torch.tensor(None)
        for criterion in self.params['training']['criterion']:
            if 'mse' in criterion:
                loss += F.mse_loss(output, ground_truth)
            if 'cross_entropy' in criterion:
                loss += F.cross_entropy(output, ground_truth)
        return loss

    def training_step(self, batch):
        """Training step"""
        data, label = batch
        # output = self(data)
        # loss = self.compute_loss(output, label)
        loss = data
        # self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch):
        """Validation step"""
        data, label = batch
        # output = self(data)
        # loss = self.compute_loss(output, label)
        loss = label
        # self.log('val_loss', loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        """Validation epoch"""
        # val_loss, num_items = 0, 0
        # for output in outputs:
        #     val_loss += output['val_loss'].sum().item()
        #     num_items += len(output['val_loss'])
        # mean_val_loss = torch.tensor(val_loss / (num_items + 1e-4))
        # self.log('val_loss', mean_val_loss)

    def configure_optimizers(self):
        """Configure optimizers"""
        if 'Adam' in self.params['training']['optimizer']:
            return torch.optim.Adam(
                params=self.parameters(),
                lr=self.params['training']['learning_rate'],
                betas=self.params['training']['betas'],
                weight_decay=self.params['training']['weight_decay'],
                eps=self.params['training']['eps'],
                amsgrad=self.params['training']['amsgrad'],
            )
        if 'SGD' in self.params['training']['optimizer']:
            return torch.optim.SGD(
                params=self.parameters(),
                lr=self.params['training']['learning_rate'],
                weight_decay=self.params['training']['weight_decay'],
            )
        raise ValueError('Invalid optimizer settings -> conf.py -> training -> optimizer')

    def train_dataloader(self):
        """Train dataloader"""
        return DataLoader(
            self.train_set,
            batch_size=self.params['training']['batch_size'],
            num_workers=self.params['meta']['num_workers'],
        )

    def val_dataloader(self):
        """Validation dataloader"""
        return DataLoader(
            self.val_set,
            batch_size=self.params['training']['batch_size'],
            num_workers=self.params['meta']['num_workers'],
        )
