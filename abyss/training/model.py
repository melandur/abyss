import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from .create_dataset import get_loader
from .create_network import get_network


class LearningRateScheduler:
    """Learning rate scheduler"""

    def __init__(self, optimizer, config: dict) -> None:
        self.optimizer = optimizer
        self.lr_start = 1e-8  # low initial learning rate
        self.lr_end = config['training']['learning_rate']
        self.warmup_epochs = config['training']['warmup']
        self.total_epochs = config['training']['max_epochs']

    def step(self, epoch: int) -> float:
        """Update optimizer learning rate for each epoch"""
        if epoch <= self.warmup_epochs:  # cosine annealing warmup
            lr = (
                self.lr_start
                + (self.lr_end - self.lr_start) * (1.0 - math.cos(epoch / self.warmup_epochs * math.pi)) / 2
            )
        else:  # poly decay
            epoch = epoch - self.warmup_epochs
            lr_current = self.optimizer.param_groups[0]['lr']
            lr = lr_current * (1.0 - epoch / (self.total_epochs - self.warmup_epochs)) ** 0.9

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def state_dict(self):
        """Return state dict"""
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.__dict__.update(state_dict)


class Model(pl.LightningModule):
    """Holds model definitions"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.net = get_network(config)
        self.criterion = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)
        self.metrics = {'dice': DiceMetric(include_background=False, reduction='mean_batch')}
        self.inferer = SlidingWindowInferer(
            roi_size=config['trainer']['patch_size'],
            sw_batch_size=4,
            overlap=0.25,
            mode='gaussian',
        )
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataset = None

    def lr_scheduler_step(self, scheduler, metric) -> None:
        """Update optimizer learning rate"""
        scheduler.step(self.current_epoch)

    def configure_optimizers(self):
        """Optimizer"""
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.config['training']['learning_rate'],
            momentum=0.99,
            weight_decay=3e-5,
            nesterov=True,
        )
        scheduler = LearningRateScheduler(optimizer, self.config)
        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward step"""
        return self.net(x)

    def training_step(self, batch: torch.Tensor) -> float:
        """Predict, loss, log, (backprop and optimizer step done by lightning)"""
        data, label = batch['image'], batch['label']
        preds = self(data)

        if len(preds.size()) - len(label.size()) == 1:  # deep supervision mode
            preds = torch.unbind(preds, dim=1)  # unbind feature maps
            factor = 2.0
            # max_epochs = self.config['training']['max_epochs']
            # epoch = self.current_epoch if self.current_epoch < 15 else 1.0
            # factor = 1 + 1000 ** math.sin(epoch / max_epochs)
            normalize_factor = sum(1.0 / (factor**i) for i in range(len(preds)))
            loss = sum(1.0 / (factor**i) * self.criterion(p, label) for i, p in enumerate(preds))  # [1, 0.5, 0.25, ...]
            loss = loss / normalize_factor
        else:  # normal mode, only last feature map is output
            loss = self.criterion(preds, label)

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, on_step=False, on_epoch=True)

    def validation_step(self, batch: torch.Tensor) -> None:
        """Predict, loss, log"""
        data, label = batch['image'], batch['label']
        pred = self.inferer(data, self.net)

        if self.config['trainer']['tta']:
            ct = 1.0
            for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                flip_inputs = torch.flip(data, dims=dims)
                flip_pred = torch.flip(self.inferer(flip_inputs, self.net), dims=dims)
                flip_pred = nn.functional.softmax(flip_pred, dim=1)
                del flip_inputs
                pred += flip_pred
                del flip_pred
                ct += 1.0
            pred = pred / ct
        else:
            pred = nn.functional.softmax(pred, dim=1)

        loss = self.criterion(pred, label)

        post_pred = AsDiscrete(argmax=True, to_onehot=pred.shape[1])
        post_label = AsDiscrete(to_onehot=pred.shape[1])

        pred = post_pred(decollate_batch(pred)[0])
        label = post_label(decollate_batch(label)[0])

        self.metrics['dice']([pred], [label])
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Log dice metric"""
        dice_metric = self.metrics['dice'].aggregate()
        mean_dice = dice_metric.mean().item()
        dice_per_label = {
            'dice_avg': mean_dice,
            'dice_ed': dice_metric[0],
            'dice_et': dice_metric[1],
            'dice_nc': dice_metric[2],
        }
        self.metrics['dice'].reset()
        self.log_dict(dice_per_label, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch: torch.Tensor) -> None:
        """Predict, loss, log"""
        data, label = batch['image'], batch['label']
        pred = self.inferer(data, self.net)
        if self.config['trainer']['tta']:
            ct = 1.0
            for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                flip_inputs = torch.flip(data, dims=dims)
                flip_pred = torch.flip(self.inferer(flip_inputs, self.net), dims=dims)
                flip_pred = nn.functional.softmax(flip_pred, dim=1)
                del flip_inputs
                pred += flip_pred
                del flip_pred
                ct += 1.0
            pred = pred / ct
        else:
            pred = nn.functional.softmax(pred, dim=1)

        post_pred = AsDiscrete(argmax=True, to_onehot=pred.shape[1])
        post_label = AsDiscrete(to_onehot=pred.shape[1])

        pred = post_pred(decollate_batch(pred)[0])
        label = post_label(decollate_batch(label)[0])

        new = torch.zeros(240, 240, 155)
        new[pred[1] == 1] = 1
        new[pred[2] == 1] = 2
        new[pred[3] == 1] = 3
        import SimpleITK as sitk

        img = sitk.GetImageFromArray(new.cpu().numpy())
        import random

        x = random.randint(0, 1000)
        sitk.WriteImage(img, f'{x}_pred.nii.gz')

        new = torch.zeros(240, 240, 155)
        new[label[1] == 1] = 1
        new[label[2] == 1] = 2
        new[label[3] == 1] = 3
        img = sitk.GetImageFromArray(new.cpu().numpy())
        sitk.WriteImage(img, f'{x}_label.nii.gz')

        self.metrics['dice']([pred], [label])

    def on_test_epoch_end(self) -> None:
        """Log dice metric"""
        dice_metric = self.metrics['dice'].aggregate()
        mean_dice = dice_metric.mean().item()
        dice_per_label = {
            'dice_avg': mean_dice,
            'dice_ed': dice_metric[0],
            'dice_et': dice_metric[1],
            'dice_nc': dice_metric[2],
        }
        self.metrics['dice'].reset()
        self.log_dict(dice_per_label, prog_bar=True, on_step=False, on_epoch=True)

    def train_dataloader(self):
        """Train dataloader"""
        return get_loader(self.config, 'train')

    def val_dataloader(self):
        """Validation dataloader"""
        return get_loader(self.config, 'val')

    def test_dataloader(self):
        """Test dataloader"""
        return get_loader(self.config, 'test')
