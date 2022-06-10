from typing import ClassVar

import torchmetrics
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger


class Trainer:
    """Based on pytorch_lightning trainer"""

    def __init__(self, config_manager: ClassVar):
        self.params = config_manager.params

        # Integrated loggers: TBoard, MLflow, Comet, Neptune, WandB
        self.logger = TensorBoardLogger(save_dir=self.params['project']['result_store_path'])

        # Define callbacks
        self.checkpoint_cb = ModelCheckpoint(
            dirpath=self.params['project']['result_store_path'], filename=self.params['project']['name']
        )

        self.early_stop_cb = EarlyStopping(
            monitor='val_loss',
            min_delta=self.params['trainer']['early_stop']['min_delta'],
            patience=self.params['trainer']['early_stop']['patience'],
            verbose=self.params['trainer']['early_stop']['verbose'],
            mode=self.params['trainer']['early_stop']['mode'],
        )

        self.model_summary_cb = RichModelSummary(self.params['trainer']['model_summary_depth'])

        self.progress_bar_cb = RichProgressBar(
            leave=True,
            theme=RichProgressBarTheme(
                description='gray82',
                progress_bar='yellow4',
                progress_bar_finished='gray82',
                progress_bar_pulse='gray82',
                batch_progress='gray82',
                time='grey82',
                processing_speed='grey82',
                metrics='grey82',
            ),
        )

        if self.params['meta']['seed']:
            seed_everything(self.params['meta']['seed'])

        torchmetrics.Metric.full_state_update = False  # will be default False in v0.1

    def __call__(self):
        return LightningTrainer(
            logger=self.logger,
            enable_checkpointing=True,
            callbacks=[self.checkpoint_cb, self.early_stop_cb, self.model_summary_cb, self.progress_bar_cb],
            default_root_dir=self.params['trainer']['default_root_dir'],
            gradient_clip_val=None,
            gradient_clip_algorithm=None,
            process_position=0,
            num_nodes=1,
            num_processes=None,
            devices=self.params['trainer']['devices'],
            gpus=self.params['trainer']['gpus'],
            auto_select_gpus=self.params['trainer']['auto_select_gpus'],
            tpu_cores=self.params['trainer']['tpu_cores'],
            ipus=None,
            log_gpu_memory=None,  # TODO: Remove in 1.7
            progress_bar_refresh_rate=None,  # TODO: remove in v1.7
            enable_progress_bar=self.params['trainer']['enable_progress_bar'],
            overfit_batches=0.0,
            track_grad_norm=-1,
            check_val_every_n_epoch=self.params['trainer']['check_val_every_n_epoch'],
            fast_dev_run=self.params['trainer']['fast_dev_run'],
            accumulate_grad_batches=None,
            max_epochs=self.params['trainer']['max_epochs'],
            min_epochs=None,
            max_steps=-1,
            min_steps=None,
            max_time=None,
            limit_train_batches=None,
            limit_val_batches=None,
            limit_test_batches=None,
            limit_predict_batches=None,
            val_check_interval=None,
            flush_logs_every_n_steps=None,
            log_every_n_steps=self.params['trainer']['log_every_n_steps'],
            accelerator=self.params['trainer']['accelerator'],
            strategy=None,
            sync_batchnorm=False,
            precision=self.params['trainer']['precision'],
            enable_model_summary=False,  # deprecated, now bey model summary callback
            weights_save_path=None,  # TODO: Remove in 1.8
            num_sanity_val_steps=2,
            resume_from_checkpoint=self.params['trainer']['resume_from_checkpoint'],
            profiler=None,
            benchmark=None,
            deterministic=self.params['trainer']['deterministic'],
            reload_dataloaders_every_n_epochs=0,
            auto_lr_find=self.params['trainer']['auto_lr_find'],
            replace_sampler_ddp=True,
            detect_anomaly=False,
            auto_scale_batch_size=False,
            prepare_data_per_node=None,
            plugins=None,
            amp_backend='native',
            amp_level=None,
            move_metrics_to_cpu=False,
            multiple_trainloader_mode='max_size_cycle',
            stochastic_weight_avg=False,
            terminate_on_nan=None,
        )
