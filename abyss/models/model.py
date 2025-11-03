import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from abyss.data.create_dataset import get_loader
from abyss.models.create_network import get_network
from abyss.losses.loss import CrossEntropyLoss, DiceCELoss, DiceCELossTopK
from abyss.engine.scheduler import WarmupPolyLRScheduler

torch.set_float32_matmul_precision('medium')


class Model(pl.LightningModule):
    """Holds model definitions"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.net = get_network(config)
        task = config['trainer']['task']

        # Select loss by name only; use built-in defaults
        loss_name = (config['training'].get('loss') or 'dice_ce').lower()

        if task == 'classification':
            self.criterion = CrossEntropyLoss()
            logger.info('Using loss: ce (classification)')
        elif task == 'segmentation':
            if loss_name == 'ce':
                self.criterion = CrossEntropyLoss()
                logger.info('Using loss: ce')
            elif loss_name == 'dice_ce_topk':
                self.criterion = DiceCELossTopK()
                logger.info('Using loss: dice_ce_topk')
            else:
                self.criterion = DiceCELoss()
                logger.info('Using loss: dice_ce')
        elif task == 'detection':
            raise NotImplementedError('Detection task is not implemented yet.')
        else:
            raise ValueError(f'Unknown task: {task}, config -> trainer -> task')

        self.metrics = {'dice': DiceMetric(reduction='none', ignore_empty=True)}

    def setup(self, stage: str) -> None:
        """Setup"""
        if stage == 'test':
            self.net.eval()

    def configure_optimizers(self):
        """Optimizer"""
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.config['training']['lr'],
            momentum=0.99,
            weight_decay=3e-5,
            nesterov=True,
        )
        scheduler = WarmupPolyLRScheduler(
            optimizer,
            base_lr=self.config['training']['lr'],
            total_steps=self.config['training']['epochs'],
            warmup_steps=self.config['training']['warmup_epochs'],
            exponent=0.9,
        )
        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward step"""
        return self.net(x)

    def on_train_epoch_start(self) -> None:
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, on_step=False, on_epoch=True)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """Predict, loss, log, (backprop and optimizer step done by lightning)"""
        data, label = batch['image'], batch['label']
        preds = self(data)

        if isinstance(preds, (list, tuple)):
            loss = 0.0
            normalize_factor = sum(1.0 / (2**i) for i in range(len(preds)))

            for idx, (pred, scale) in enumerate(zip(preds, [1, 0.5, 0.25])):
                scaled_target = F.interpolate(label, scale_factor=scale, mode='nearest')
                loss += (1.0 / (2**idx)) * self.criterion(pred, scaled_target)

            loss = loss / normalize_factor
        else:
            loss = self.criterion(preds, label)

        self.log('loss_train', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor) -> None:
        """Predict, loss, log"""
        data, label = batch['image'], batch['label']

        preds = self(data)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        loss = self.criterion(preds, label)
        pred = nn.functional.sigmoid(preds)
        post_pred = AsDiscrete(threshold=0.5)
        pred = post_pred(decollate_batch(pred)[0])
        label = decollate_batch(label)[0]

        self.metrics['dice']([pred], [label])
        self.log('loss_val', loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Log dice metric"""
        dice_metric = self.metrics['dice'].aggregate()
        label_classes = self.config['trainer']['label_classes']
        dice_per_channel = torch.nanmean(dice_metric, dim=0)
        metric_log = {'dice_avg': torch.mean(dice_per_channel, dim=0)}
        for idx, label_class in enumerate(label_classes):
            metric_log[f'dice_{label_class}'] = dice_per_channel[idx]
        self.metrics['dice'].reset()
        self.log_dict(metric_log, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch: torch.Tensor) -> None:
        """Predict, loss, log"""
        data, label = batch['image'], batch['label']
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

        self.metrics['dice']([pred], [label])

        pred = torch.argmax(pred, dim=0)
        label = torch.argmax(label, dim=0)
        import random

        import SimpleITK as sitk

        path = os.path.join(self.config['project']['results_path'], 'inference')
        os.makedirs(path, exist_ok=True)
        x = random.randint(0, 1000)
        img = sitk.GetImageFromArray(pred.cpu().numpy())
        sitk.WriteImage(img, os.path.join(path, f'{x}_pred.nii.gz'))

        img = sitk.GetImageFromArray(label.cpu().numpy())
        sitk.WriteImage(img, os.path.join(path, f'{x}_label.nii.gz'))

    def on_test_epoch_end(self) -> None:
        """Log dice metric"""
        dice_metric = self.metrics['dice'].aggregate()
        dice_per_channel = torch.nanmean(dice_metric, dim=0)
        metric_log = {
            'dice_avg': torch.mean(dice_per_channel, dim=0),
            'dice_tc': dice_per_channel[0],
            'dice_wt': dice_per_channel[1],
            'dice_en': dice_per_channel[2],
        }
        self.metrics['dice'].reset()
        self.log_dict(metric_log, prog_bar=True, on_step=False, on_epoch=True)

    def train_dataloader(self):
        """Train dataloader"""
        return get_loader(self.config, 'train')

    def val_dataloader(self):
        """Validation dataloader"""
        return get_loader(self.config, 'val')

    def test_dataloader(self):
        """Test dataloader"""
        return get_loader(self.config, 'test')

