from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from conf import params
from src.net import Net

if __name__ == '__main__':
    net = Net()

    # Define logger, TBoard, MLflow, Comet, Neptune, WandB
    logger = TensorBoardLogger(save_dir=params['project']['result_store_path'])

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=params['project']['result_store_path'],
                                          filename=params['project']['name'])
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.00,
                                        patience=3,
                                        verbose=False,
                                        mode='max')

    # Used defined train seed
    if params['training']['seed']:
        seed_everything(params['training']['seed'])

    # initialise Lightning's trainer, default values if not set
    trainer = Trainer(
        logger=logger,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=0.0,
        process_position=0,
        num_nodes=1,
        num_processes=1,
        gpus=[0],
        auto_select_gpus=False,
        tpu_cores=None,
        log_gpu_memory=None,
        progress_bar_refresh_rate=None,
        overfit_batches=0.0,
        track_grad_norm=-1,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        accumulate_grad_batches=1,
        max_epochs=None,
        min_epochs=None,
        max_steps=None,
        min_steps=None,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        limit_predict_batches=1.0,
        val_check_interval=1.0,
        flush_logs_every_n_steps=100,
        log_every_n_steps=50,
        accelerator=None,
        sync_batchnorm=False,
        precision=32,
        weights_summary='top',
        weights_save_path=None,
        num_sanity_val_steps=2,
        truncated_bptt_steps=None,
        resume_from_checkpoint=None,
        profiler=None,
        benchmark=False,
        deterministic=False,
        reload_dataloaders_every_epoch=False,
        auto_lr_find=False,
        replace_sampler_ddp=True,
        terminate_on_nan=False,
        auto_scale_batch_size=False,
        prepare_data_per_node=True,
        plugins=None,
        amp_backend='native',
        amp_level='O2',
        distributed_backend=None,
        automatic_optimization=None,
        move_metrics_to_cpu=False,
        multiple_trainloader_mode='max_size_cycle',
        stochastic_weight_avg=False
    )

    # train
    trainer.fit(net)
