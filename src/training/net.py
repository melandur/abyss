import os
import glob
import torch
import pytorch_lightning
from monai import data, inferers, losses, metrics, transforms, utils

from src.training.net_architecture import net_architecture
from src.training.augmentation import train_transform, val_transform


class Net(pytorch_lightning.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self._model = net_architecture
        # TODO: Native multi loss
        self.loss_function = losses.DiceLoss(to_onehot_y=False,
                                             sigmoid=True,
                                             squared_pred=True)

        self.post_pred = transforms.AsDiscrete(argmax=True,
                                               to_onehot=True,
                                               n_classes=self.params['training']['n_classes'])
        self.post_label = transforms.AsDiscrete(to_onehot=False,
                                                n_classes=self.params['training']['n_classes'])

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        data_path = os.path.join(self.params['project']['dataset_store_path'])
        train_images = sorted(glob.glob(os.path.join(data_path, 'imagesTr', '*.nii.gz')))
        train_labels = sorted(glob.glob(os.path.join(data_path, 'labelsTr', '*.nii.gz')))
        data_dicts = [{'image': image_name, 'label': label_name}
                      for image_name, label_name in zip(train_images, train_labels)]

        # TODO: Hardcoded stuff
        train_files, val_files = data_dicts[:-9], data_dicts[-9:]

        # set deterministic training for reproducibility
        utils.set_determinism(seed=self.params['dataset']['seed'])

        if self.params['dataset']['use_cache']:
            self.train_ds = data.CacheDataset(data=train_files,
                                              transform=train_transform,
                                              cache_rate=self.params['dataset']['cache_rate'],
                                              num_workers=self.params['dataset']['num_workers'])

            self.val_ds = data.CacheDataset(data=val_files,
                                            transform=val_transform,
                                            cache_rate=self.params['dataset']['cache_rate'],
                                            num_workers=self.params['dataset']['num_workers'])
        else:
            self.train_ds = data.Dataset(data=train_files, transform=train_transform)
            self.val_ds = data.Dataset(data=val_files, transform=val_transform)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_ds,
                                                   batch_size=self.params['training']['batch_size'],
                                                   shuffle=True,
                                                   num_workers=self.params['training']['num_workers'],
                                                   collate_fn=data.list_data_collate)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_ds,
                                                 batch_size=self.params['training']['batch_size'],
                                                 num_workers=self.params['training']['num_workers'])
        return val_loader

    def configure_optimizers(self):
        optimizer = None
        if 'Adam' in self.params['training']['optimizer']:
            optimizer = torch.optim.Adam(params=self._model.parameters(),
                                         lr=self.params['training']['learning_rate'],
                                         betas=self.params['training']['betas'],
                                         weight_decay=self.params['training']['weight_decay'],
                                         eps=self.params['training']['eps'],
                                         amsgrad=self.params['training']['amsgrad'])

        elif 'SGD' in self.params['training']['optimizer']:
            optimizer = torch.optim.SGD(params=self._model.parameters(),
                                        lr=self.params['training']['learning_rate'],
                                        weight_decay=self.params['training']['weight_decay'])

        assert optimizer is not None, 'Invalid optimizer settings in conf.py: training, optimizer'
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        roi_size = (160, 160, 160)
        sw_batch_size = 1
        outputs = inferers.sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        outputs = self.post_pred(outputs)
        labels = self.post_label(labels)
        value = metrics.compute_meandice(y_pred=outputs,
                                         y=labels,
                                         include_background=False)
        self.log('val_loss', loss)
        self.log('val_dice', value)

    def validation_epoch_end(self, outputs):
        val_dice, val_loss, num_items = 0, 0, 0
        for output in outputs:
            val_dice += output['val_dice'].sum().item()
            val_loss += output['val_loss'].sum().item()
            num_items += len(output['val_dice'])
        mean_val_dice = torch.tensor(val_dice / (num_items + 1e-4))
        mean_val_loss = torch.tensor(val_loss / (num_items + 1e-4))
        self.log('val_dice',  mean_val_dice)
        self.log('val_loss', mean_val_loss)
