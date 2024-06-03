import math
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion

from .create_dataset import get_loader
from .create_network import get_network


class Model(pl.LightningModule):
    """Holds model definitions"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.net = get_network(config)
        self.criterion = DiceCELoss(sigmoid=True)
        self.metrics = {'dice': DiceMetric(reduction='mean_channel')}
        self.inferer = SlidingWindowInferer(
            roi_size=config['trainer']['patch_size'],
            sw_batch_size=4,
            overlap=0.5,
            mode='gaussian',
        )
        self.ds_factor = 2.0

    def configure_optimizers(self):
        """Optimizer"""
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.config['training']['learning_rate'],
            momentum=0.99,
            weight_decay=3e-5,
            nesterov=True,
        )
        total_epochs = self.config['training']['max_epochs']
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / total_epochs) ** 0.9)
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]) -> None:
        """LR Scheduler step"""
        if self.trainer.global_step > self.config['training']['warmup_steps']:
            scheduler.step()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None) -> None:
        """Optimizer step with lr warmup"""
        optimizer.step(closure=optimizer_closure)

        end_lr = self.config['training']['learning_rate']
        warmup_steps = self.config['training']['warmup_steps']

        if self.trainer.global_step < warmup_steps:
            lr = end_lr * (1.0 - math.cos(self.trainer.global_step / warmup_steps * math.pi)) / 2.0
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward step"""
        return self.net(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """Predict, loss, log, (backprop and optimizer step done by lightning)"""
        data, label = batch['image'], batch['label']
        preds = self(data)

        if len(preds.size()) - len(label.size()) == 1:  # deep supervision mode
            preds = torch.unbind(preds, dim=1)  # unbind feature maps
            loss = 0.0
            normalize_factor = sum(1.0 / (self.ds_factor**i) for i in range(len(preds)))
            for idx, pred in enumerate(preds):
                loss += 1.0 / (self.ds_factor ** idx) * self.criterion(pred, label)
            loss = loss / normalize_factor
        else:  # only last feature map is output is used
            loss = self.criterion(preds, label)

        self.log('loss_train', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """None"""
        if self.trainer.global_step > self.config['training']['warmup_steps']:  # after warmup deep supervision decay
            max_epochs = self.config['training']['max_epochs']
            self.ds_factor = 1 + 1000 ** math.sin(self.current_epoch / max_epochs)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, on_step=False, on_epoch=True)

    def validation_step(self, batch: torch.Tensor) -> None:
        """Predict, loss, log"""
        data, label = batch['image'], batch['label']
        pred = self.inferer(data, self.net)

        if self.config['trainer']['tta']:
            ct = 1.0
            for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                flip_inputs = torch.flip(data, dims=dims)
                flip_pred = self.inferer(flip_inputs, self.net)
                flip_pred = torch.flip(flip_pred, dims=dims)
                del flip_inputs
                pred += flip_pred
                del flip_pred
                ct += 1.0
            pred = pred / ct

        label = label[:, 1:, ...]  # remove background
        pred = pred[:, 1:, ...]

        loss = self.criterion(pred, label)
        post_pred = AsDiscrete(threshold=0.5)
        pred = nn.functional.sigmoid(pred)
        pred = post_pred(decollate_batch(pred)[0])
        label = decollate_batch(label)[0]

        self.metrics['dice'](pred, label)
        self.log('loss_val', loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Log dice metric"""
        dice_metric = self.metrics['dice'].aggregate()
        label_classes = self.config['trainer']['label_classes']
        dice_per_label = {'dice_avg': dice_metric.mean()}
        for idx, label_class in enumerate(label_classes, start=-1):
            if label_class == 'background':
                continue
            dice_per_label[f'dice_{label_class}'] = dice_metric[idx]
        self.metrics['dice'].reset()
        self.log_dict(dice_per_label, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch: torch.Tensor) -> None:
        """Predict, loss, log"""
        data, label = batch['image'], batch['label']
        label = label.cpu()
        pred = self.inferer(data, self.net)

        if self.config['trainer']['tta']:
            ct = 1.0
            for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                flip_inputs = torch.flip(data, dims=dims)
                flip_pred = torch.flip(self.inferer(flip_inputs, self.net), dims=dims)
                del flip_inputs
                pred += flip_pred
                del flip_pred
                ct += 1.0
            pred = pred / ct

        pred = nn.functional.sigmoid(pred)
        post_pred = AsDiscrete(threshold=0.5)

        pred = post_pred(decollate_batch(pred)[0])
        label = decollate_batch(label)[0]

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
            'dice_ed': dice_metric[0].mean().item(),
            'dice_nc': dice_metric[1].mean().item(),
            'dice_en': dice_metric[2].mean().item(),
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
