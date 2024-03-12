import os

import torch
import torchmetrics
from loguru import logger

from abyss.config import ConfigManager


class Trainer(ConfigManager):
    """Based on pytorch_lightning trainer"""

    def __init__(self, model, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)

        self.model = model
        self.log = None
        self.optimizer = None
        self.criterion = None
        self.checkpoint = None
        self.early_stop = None
        self.model_summary = None
        self.progress_bar = None
        self.seed = None
        self.num_threads = None

    def __call__(self):

        self.model.setup('fit')

        # torch specific handling
        pytorch_train_dataset = pymia_torch.PytorchDatasetAdapter(train_dataset)
        train_loader = torch_data.dataloader.DataLoader(pytorch_train_dataset, batch_size=16, shuffle=True)

        pytorch_valid_dataset = pymia_torch.PytorchDatasetAdapter(valid_dataset)
        valid_loader = torch_data.dataloader.DataLoader(pytorch_valid_dataset, batch_size=16, shuffle=False)

        u_net = unet.UNetModel(ch_in=2, ch_out=6, n_channels=16, n_pooling=3).to(device)

        print(u_net)

        optimizer = optim.Adam(u_net.parameters(), lr=1e-3)
        train_batches = len(train_loader)

    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0

            with tqdm(self.train_loader, unit="batch") as tepoch:
                for data, target in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())

                    # Log images and model weights to TensorBoard
                    self.tensorboard_logger.experiment.add_image('input_images', make_grid(data), epoch)
                    self.tensorboard_logger.experiment.add_histogram('model_weights', self.model.state_dict(), epoch)

                    # Log model predictions and ground truth to BIAS
                    self.bias_logger.experiment.log_predictions(output, target)

            val_loss = self.validate(epoch)
            self.logger.info(
                f"Epoch {epoch} | Train Loss: {train_loss / len(self.train_loader)} | Val Loss: {val_loss}")

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            with tqdm(self.val_loader, unit="batch") as tepoch:
                for data, target in tepoch:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    val_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())

                    # Log model predictions and ground truth to BIAS
                    self.bias_logger.experiment.log_predictions(output, target)

        # Log validation loss to TensorBoard
        self.tensorboard_logger.experiment.add_scalar('val_loss', val_loss / len(self.val_loader), epoch)

        return val_loss / len(self.val_loader)


    def test(self):
        self.model.setup('test')
        test_loss = 0.0

        with torch.no_grad():
            with tqdm(self.test_loader, unit="batch") as tepoch:
                for data, target in tepoch:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    test_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())

                    # Log model predictions and ground truth to BIAS
                    self.bias_logger.experiment.log_predictions(output, target)

        self.logger.info(f"Test Loss: {test_loss / len(self.test_loader)}")
        return test_loss / len(self.test_loader)


#
# class Trainer(ConfigManager):
#     """Based on pytorch_lightning trainer"""
#
#     def __init__(self, **kwargs) -> None:
#         super().__init__()
#         self._shared_state.update(kwargs)
#
#         # Integrated loggers: TBoard, MLflow, Comet, Neptune, WandB
#         results_store_path = self.params['project']['result_store_path']
#         self.logger = TensorBoardLogger(save_dir=results_store_path)
#         logger.info(f'tensorboard --logdir={os.path.join(results_store_path, "lightning_logs")}')
#
#         # Define callbacks
#         self.checkpoint_cb = ModelCheckpoint(
#             dirpath=os.path.join(self.params['project']['result_store_path'], 'checkpoints'),
#             filename=self.params['project']['name'] + '_best_{epoch:02d}_{val_loss:.2f}',
#             save_last=True,
#             monitor='val_loss',
#         )
#
#         self.early_stop_cb = EarlyStopping(
#             monitor='val_loss',
#             min_delta=self.params['trainer']['early_stop']['min_delta'],
#             patience=self.params['trainer']['early_stop']['patience'],
#             verbose=self.params['trainer']['early_stop']['verbose'],
#             mode=self.params['trainer']['early_stop']['mode'],
#         )
#
#         self.model_summary_cb = RichModelSummary(self.params['trainer']['model_summary_depth'])
#
#         self.progress_bar_cb = RichProgressBar(
#             leave=True,
#             theme=RichProgressBarTheme(
#                 description='gray82',
#                 progress_bar='yellow4',
#                 progress_bar_finished='gray82',
#                 progress_bar_pulse='gray82',
#                 batch_progress='gray82',
#                 time='grey82',
#                 processing_speed='grey82',
#                 metrics='grey82',
#                 metrics_format=".3f"
#             ),
#         )
#
#         if self.params['meta']['seed']:
#             seed_everything(self.params['meta']['seed'])
#
#         torch.set_num_threads(self.params['meta']['num_workers'])
#         torchmetrics.Metric.full_state_update = False  # will be default False in v0.1
#
#     def __call__(self) -> LightningTrainer:
#         return LightningTrainer(
#             accelerator=self.params['trainer']['accelerator'],  # Union[str, Accelerator]
#             strategy="auto",  # Union[str, Strategy]
#             devices=self.params['trainer']['devices'],  # Union[List[int], str, int]
#             num_nodes=1,  # int
#             precision=None,  # Optional[_PRECISION_INPUT]
#             logger=self.logger,  # Optional[Union[Logger, Iterable[Logger], bool]]
#             callbacks=[
#                     self.checkpoint_cb,
#                     self.early_stop_cb,
#                     # self.model_summary_cb,
#                     # self.progress_bar_cb,
#                 ],  # Optional[Union[List[Callback], Callback]]
#             fast_dev_run=self.params['trainer']['fast_dev_run'],  # Union[int, bool]
#             max_epochs=None,  # Optional[int]
#             min_epochs=None,  # Optional[int]
#             max_steps=-1,  # int
#             min_steps=None,  # Optional[int]
#             max_time=None,  # Optional[Union[str, timedelta, Dict[str, int]]]
#             limit_train_batches=None,  # Optional[Union[int, float]]
#             limit_val_batches=None,  # Optional[Union[int, float]]
#             limit_test_batches=None,  # Optional[Union[int, float]]
#             limit_predict_batches=None,  # Optional[Union[int, float]]
#             overfit_batches=0.0,  # Union[int, float]
#             val_check_interval=None,  # Optional[Union[int, float]]
#             check_val_every_n_epoch=self.params['trainer']['check_val_every_n_epoch'],  # Optional[int]
#             num_sanity_val_steps=None,  # Optional[int]
#             log_every_n_steps=self.params['trainer']['log_every_n_steps'],  #  Optional[int]
#             enable_checkpointing=True,  # Optional[bool]
#             enable_progress_bar=self.params['trainer']['enable_progress_bar'],  # Optional[bool]
#             enable_model_summary=True,  # Optional[bool]
#             accumulate_grad_batches=1,  # int
#             gradient_clip_val=None,  # Optional[Union[int, float]]
#             gradient_clip_algorithm=None,  # Optional[str]
#             deterministic=self.params['trainer']['deterministic'],  # Optional[Union[bool, _LITERAL_WARN]]
#             benchmark=None,  # Optional[bool]
#             inference_mode=True,  # bool
#             use_distributed_sampler=True,  # bool
#             profiler=None,  # Optional[Union[Profiler, str]]
#             detect_anomaly=False,  # bool
#             barebones=False,  # bool
#             plugins=None,  # Optional[Union[_PLUGIN_INPUT, List[_PLUGIN_INPUT]]]
#             sync_batchnorm=False,  # bool
#             reload_dataloaders_every_n_epochs=0,  # int
#             default_root_dir=self.params['trainer']['default_root_dir']  # Optional[_PATH]
#         )
