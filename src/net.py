import os
import glob
import torch
import numpy as np
import pytorch_lightning
import matplotlib.pyplot as plt
from monai import data, inferers, losses, metrics, transforms, utils

from conf import params
from src.net_architecture import net_architecture


# TODO: Highly specific class, move to transform
class ConvertToMultiChannelBasedOnBratsClassesd(transforms.MapTransform):
    '''
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    '''

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = net_architecture
        self.loss_function = losses.DiceLoss(to_onehot_y=False,
                                             sigmoid=True,
                                             squared_pred=True)
        self.post_pred = transforms.AsDiscrete(argmax=True,
                                               to_onehot=True,
                                               n_classes=params['training']['n_classes'])
        self.post_label = transforms.AsDiscrete(to_onehot=False,
                                                n_classes=params['training']['n_classes'])

    def forward(self, x):
        return self._model(x)

    def configure_callbacks(self):
        """Here goes the early stopping"""

    def prepare_data(self):
        # set up the correct data path
        train_images = sorted(glob.glob(
            os.path.join(params['project']['dataset_store_path'], params['data']['challenge'], 'imagesTr', '*.nii.gz')))
        train_labels = sorted(glob.glob(
            os.path.join(params['project']['dataset_store_path'], params['data']['challenge'], 'labelsTr', '*.nii.gz')))
        data_dicts = [{'image': image_name, 'label': label_name}
                      for image_name, label_name in zip(train_images, train_labels)]
        train_files, val_files = data_dicts[:-9], data_dicts[-9:]
        #
        # set deterministic training for reproducibility
        utils.set_determinism(seed=params['data']['seed'])

        # define the data transforms
        # TODO: Add your data augmentation check out monai.transforms
        train_transform = transforms.Compose(
            [
                # load 4 Nifti images and stack them together
                transforms.LoadImaged(keys=['image', 'label']),
                transforms.AsChannelFirstd(keys='image'),
                ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
                transforms.Spacingd(keys=['image', 'label'],
                                    pixdim=(1.5, 1.5, 2.0),
                                    mode=('bilinear', 'nearest')),
                transforms.Orientationd(keys=['image', 'label'], axcodes='RAS'),
                transforms.RandSpatialCropd(keys=['image', 'label'], roi_size=[128, 128, 64], random_size=False),
                transforms.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
                transforms.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys='image', factors=0.1, prob=0.5),
                transforms.RandShiftIntensityd(keys='image', offsets=0.1, prob=0.5),
                transforms.ToTensord(keys=['image', 'label']),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=['image', 'label']),
                transforms.AsChannelFirstd(keys='image'),
                ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
                transforms.Spacingd(keys=['image', 'label'],
                                    pixdim=(1.5, 1.5, 2.0),
                                    mode=('bilinear', 'nearest')),
                transforms.Orientationd(keys=['image', 'label'], axcodes='RAS'),
                transforms.CenterSpatialCropd(keys=['image', 'label'], roi_size=[128, 128, 64]),
                transforms.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=['image', 'label']),
            ]
        )

        if params['data']['use_cache']:
            # we use cached datasets - these are 10x faster than regular datasets
            self.train_ds = data.CacheDataset(data=train_files,
                                              transform=train_transform,
                                              cache_rate=params['data']['cache_rate'],
                                              num_workers=params['data']['num_workers'])

            self.val_ds = data.CacheDataset(data=val_files,
                                            transform=val_transform,
                                            cache_rate=params['data']['cache_rate'],
                                            num_workers=params['data']['num_workers'])
        else:
            self.train_ds = data.Dataset(data=train_files, transform=train_transform)
            self.val_ds = data.Dataset(data=val_files, transform=val_transform)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_ds,
                                                   batch_size=params['training']['batch_size'],
                                                   shuffle=True,
                                                   num_workers=params['training']['num_workers'],
                                                   collate_fn=data.list_data_collate)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_ds,
                                                 batch_size=params['training']['batch_size'],
                                                 num_workers=params['training']['num_workers'])
        return val_loader

    def configure_optimizers(self):
        if 'Adam' in params['training']['optimizer']:

            optimizer = torch.optim.Adam(params=self._model.parameters(),
                                         lr=params['training']['learning_rate'],
                                         betas=params['training']['betas'],
                                         weight_decay=params['training']['weight_decay'],
                                         eps=params['training']['eps'],
                                         amsgrad=params['training']['amsgrad'])

        elif 'SGD' in params['training']['optimizer']:
            optimizer = torch.optim.SGD(params=self._model.parameters(),
                                        lr=params['training']['learning_rate'],
                                        weight_decay=params['training']['weight_decay'])

        else:
            print('Invalid optimizer settings in conf.py: training, optimizer')
            exit(1)
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
        print(value, loss)
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
