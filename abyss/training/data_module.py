import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader

# from abyss.training.augmentation import DataAugmentation
from abyss.training.models import UNet


class DataModule(pl.LightningModule):
    """Net definition"""

    def __init__(self, _config_manager):
        super().__init__()
        self.config_manager = _config_manager
        self.params = _config_manager.params
        self.val_ds = None
        self.train_ds = None

        self.model = UNet
        # TODO: Make native a multiloss available
        # self.loss_function = losses.DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        # self.post_pred = transforms.AsDiscrete(
        #     argmax=True,
        #     to_onehot=True,
        #     n_classes=self.params['training']['n_classes'],
        # )
        # self.post_label = transforms.AsDiscrete(to_onehot=False, n_classes=self.params['training']['n_classes'])

    def forward(self, x):
        return self.model(x)

    # def prepare_data(self):
    #     train_files = self.config_manager.get_path_memory('train_dataset_paths')
    #     val_files = self.config_manager.get_path_memory('val_dataset_paths')
    #     # utils.set_determinism(seed=self.params['dataset']['seed'])  # set training deterministic
    #
    #     data_augmentation = DataAugmentation()
    #     if self.params['dataset']['use_cache']:
    #         self.train_ds = data.CacheDataset(
    #             data=train_files,
    #             transform=data_augmentation.train_transform,
    #             cache_rate=self.params['dataset']['cache_rate'],
    #             num_workers=self.params['dataset']['num_workers'],
    #         )
    #
    #         self.val_ds = data.CacheDataset(
    #             data=val_files,
    #             transform=data_augmentation.val_transform,
    #             cache_rate=self.params['dataset']['cache_rate'],
    #             num_workers=self.params['dataset']['num_workers'],
    #         )
    #     else:
    #         self.train_ds = data.Dataset(data=train_files, transform=data_augmentation.train_transform)
    #         self.val_ds = data.Dataset(data=val_files, transform=data_augmentation.val_transform)

    # # pick one image from DecathlonDataset to visualize and check the 4 channels
    # print(f"image shape: {self.val_ds[0]['data'].shape}")
    # plt.figure("image", (24, 6))
    # for i in range(4):
    #     plt.subplot(1, 4, i + 1)
    #     plt.title(f"image channel {i}")
    #     plt.imshow(self.val_ds[1]["image"][i, :, :, 20].detach().cpu(), cmap="gray")
    # plt.show()
    # # also visualize the 3 channels label corresponding to this image
    # print(f"label shape: {self.val_ds[0]['label'].shape}")
    # plt.figure("label", (18, 6))
    # for i in range(3):
    #     plt.subplot(1, 3, i + 1)
    #     plt.title(f"label channel {i}")
    #     plt.imshow(self.val_ds[0]["label"][i, :, :, 20].detach().cpu())
    # plt.show()

    def training_step(self, batch):
        """Training step"""
        # images, labels = batch['data'], batch['label']
        # output = self.forward(images)
        # loss = self.loss_function(output, labels)
        # self.log('train_loss', loss.item())
        # return loss

    def validation_step(self, batch):
        """Validation step"""
        # images, labels = batch['data'], batch['label']
        # roi_size = (160, 160, 160)
        # sw_batch_size = 1
        # outputs = inferers.sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        # loss = self.loss_function(outputs, labels)
        # outputs = self.post_pred(outputs)
        # labels = self.post_label(labels)
        # value = metrics.compute_meandice(y_pred=outputs, y=labels, include_background=False)
        # self.log('val_loss', loss)
        # self.log('val_dice', value)

    def validation_epoch_end(self, outputs):
        """Validation epoch"""
        # val_dice, val_loss, num_items = 0, 0, 0
        # for output in outputs:
        #     val_dice += output['val_dice'].sum().item()
        #     val_loss += output['val_loss'].sum().item()
        #     num_items += len(output['val_dice'])
        # mean_val_dice = torch.tensor(val_dice / (num_items + 1e-4))
        # mean_val_loss = torch.tensor(val_loss / (num_items + 1e-4))
        # self.log('val_dice', mean_val_dice)
        # self.log('val_loss', mean_val_loss)

    def configure_optimizers(self):
        """Configure optimizers"""
        if 'Adam' in self.params['training']['optimizer']:
            return torch.optim.Adam(
                params=self._model.parameters(),
                lr=self.params['training']['learning_rate'],
                betas=self.params['training']['betas'],
                weight_decay=self.params['training']['weight_decay'],
                eps=self.params['training']['eps'],
                amsgrad=self.params['training']['amsgrad'],
            )

        elif 'SGD' in self.params['training']['optimizer']:
            return torch.optim.SGD(
                params=self._model.parameters(),
                lr=self.params['training']['learning_rate'],
                weight_decay=self.params['training']['weight_decay'],
            )
        else:
            raise ValueError('Invalid optimizer settings in conf.py: training, optimizer')

    def train_dataloader(self):
        """Train dataloader"""
        return DataLoader(
            self.train_ds,
            batch_size=self.params['training']['batch_size'],
            shuffle=True,
            num_workers=self.params['training']['num_workers'],
            # collate_fn=torch.utils.data.list_data_collate,
        )

    def val_dataloader(self):
        """Validation dataloader"""
        return DataLoader(
            self.val_ds,
            batch_size=self.params['training']['batch_size'],
            num_workers=self.params['training']['num_workers'],
        )


