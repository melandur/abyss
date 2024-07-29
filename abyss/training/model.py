import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch.optim.lr_scheduler import LambdaLR

from .create_dataset import get_loader
from .create_network import get_network
from .sliding_window import sliding_window_inference


class Model(pl.LightningModule):
    """Holds model definitions"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.net = get_network(config)
        self.criterion = DiceCELoss(sigmoid=True, batch=True, squared_pred=True)
        self.metrics = {'dice': DiceMetric(reduction='none', ignore_empty=True)}
        self.infi = SlidingWindowInferer(
            roi_size=self.config['trainer']['patch_size'], sw_batch_size=1, overlap=0.5, mode='gaussian'
        )

    def setup(self, stage: str) -> None:
        """Setup"""
        if stage == 'test':
            results_files = os.listdir(self.config['project']['results_path'])
            best_ckpt_file = [file for file in results_files if 'best' in file and file.endswith('.ckpt')]
            if len(best_ckpt_file) == 1:
                best_ckpt_path = os.path.join(self.config['project']['results_path'], best_ckpt_file[0])
            else:
                raise FileNotFoundError(f'No best checkpoint found in -> {self.config["project"]["results_path"]}')

            checkpoint = torch.load(best_ckpt_path)
            weights = checkpoint['state_dict']
            for key in list(weights.keys()):
                if 'net.' in key:
                    new_key = key.replace('net.', '')
                    weights[new_key] = weights.pop(key)
                if 'criterion.' in key:
                    weights.pop(key)

            self.net.load_state_dict(weights)
            self.net.eval()

    def configure_optimizers(self):
        """Optimizer"""
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=1e-2,
            momentum=0.99,
            weight_decay=3e-5,
            nesterov=True,
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / 1000) ** 0.9)
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

        if len(preds.size()) - len(label.size()) == 1:  # deep supervision mode
            preds = torch.unbind(preds, dim=1)  # unbind feature maps
            loss = 0.0
            normalize_factor = sum(1.0 / (2**i) for i in range(len(preds)))
            for idx, pred in enumerate(preds):
                loss += 1.0 / (2**idx) * self.criterion(pred, label)
            loss = loss / normalize_factor
        else:  # only last feature map is output is used
            loss = self.criterion(preds, label)

        self.log('loss_train', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor) -> None:
        """Predict, loss, log"""
        data, label = batch['image'], batch['label']
        # pred = sliding_window_inference(data, self.config['trainer']['patch_size'], self.net)
        pred = self.infi(data, self.net)

        if self.config['trainer']['tta']:
            ct = 1.0
            for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                flip_inputs = torch.flip(data, dims=dims)
                # flip_pred = sliding_window_inference(flip_inputs, self.config['trainer']['patch_size'], self.net)
                flip_pred = self.infi(flip_inputs, self.net)
                flip_pred = torch.flip(flip_pred, dims=dims)
                del flip_inputs
                pred += flip_pred
                del flip_pred
                ct += 1.0
            pred = pred / ct

        loss = self.criterion(pred, label)
        pred = nn.functional.sigmoid(pred)
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
