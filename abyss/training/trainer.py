from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class Trainer:
    """Based on pytorch_lightning trainer"""

    def __init__(self, config_manager):
        params = config_manager.params

        # Integrated loggers: TBoard, MLflow, Comet, Neptune, WandB
        self.logger = TensorBoardLogger(save_dir=params['project']['result_store_path'])

        # Define callbacks
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=params['project']['result_store_path'], filename=params['project']['name']
        )

        self.early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=params['training']['early_stop']['min_delta'],
            patience=params['training']['early_stop']['patience'],
            verbose=params['training']['early_stop']['verbose'],
            mode=params['training']['early_stop']['mode'],
        )

        # Used defined train seed
        if params['meta']['seed']:
            seed_everything(params['meta']['seed'])

    def __call__(self):
        # initialise Lightning's trainer, default values if not specific set in conf set
        # TODO: Link to config settings
        return LightningTrainer(
            logger=self.logger,
            checkpoint_callback=True,
            callbacks=[self.checkpoint_callback, self.early_stop_callback],
            default_root_dir=None,
            gradient_clip_val=None,
            gradient_clip_algorithm=None,
            process_position=0,
            num_nodes=1,
            num_processes=None,
            devices=None,
            gpus=None,
            auto_select_gpus=False,
            tpu_cores=None,
            ipus=None,
            log_gpu_memory=None,  # TODO: Remove in 1.7
            progress_bar_refresh_rate=None,  # TODO: remove in v1.7
            enable_progress_bar=True,
            overfit_batches=0.0,
            track_grad_norm=-1,
            check_val_every_n_epoch=1,
            fast_dev_run=False,
            accumulate_grad_batches=None,
            max_epochs=None,
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
            log_every_n_steps=50,
            accelerator=None,
            strategy=None,
            sync_batchnorm=False,
            precision=32,
            enable_model_summary=True,
            weights_summary="top",
            weights_save_path=None,  # TODO: Remove in 1.8
            num_sanity_val_steps=2,
            resume_from_checkpoint=None,
            profiler=None,
            benchmark=None,
            deterministic=None,
            reload_dataloaders_every_n_epochs=0,
            auto_lr_find=False,
            replace_sampler_ddp=True,
            detect_anomaly=False,
            auto_scale_batch_size=False,
            prepare_data_per_node=None,
            plugins=None,
            amp_backend="native",
            amp_level=None,
            move_metrics_to_cpu=False,
            multiple_trainloader_mode="max_size_cycle",
            stochastic_weight_avg=False,
            terminate_on_nan=None,
        )
