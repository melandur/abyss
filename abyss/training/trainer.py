import os

import torch
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger


def get_trainer(config: dict) -> LightningTrainer:
    results_path = config['project']['results_path']
    logger = TensorBoardLogger(save_dir=results_path, name='logs')  # TBoard, MLflow, Comet, Neptune, WandB
    log_path = os.path.join(results_path, 'logs')
    print(f'tensorboard --logdir={log_path}')

    # swa = StochasticWeightAveraging(swa_lrs=1e-2, swa_epoch_start=0.001)

    early_stop_cb = EarlyStopping(
        monitor='loss_val',
        min_delta=config['training']['early_stop']['min_delta'],
        patience=config['training']['early_stop']['patience'],
        verbose=True,
        mode=config['training']['early_stop']['mode'],
    )

    model_checkpoint_cb = ModelCheckpoint(
        monitor='loss_val',
        dirpath=results_path,
        filename='best-{epoch:02d}-{loss_val:.2f}',
        save_last=True,
        save_top_k=1,
        mode='min',
    )

    progress_bar_cb = RichProgressBar(
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

    if config['training']['seed']:
        seed_everything(config['training']['seed'])

    torch.set_num_threads(config['training']['num_workers'])

    trainer = LightningTrainer(
        accelerator='gpu',
        strategy='auto',
        devices=[0],
        num_nodes=1,
        precision='16-mixed',  # '16-mixed', '16-false', '32'
        logger=logger,
        callbacks=[
            early_stop_cb,
            progress_bar_cb,
            # swa,
        ],
        fast_dev_run=config['training']['fast_dev_run'],
        max_epochs=config['training']['max_epochs'],
        min_epochs=None,
        max_steps=-1,
        min_steps=None,
        max_time=None,
        limit_train_batches=None,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        overfit_batches=0.0,
        val_check_interval=None,
        check_val_every_n_epoch=config['training']['check_val_every_n_epoch'],
        num_sanity_val_steps=None,
        log_every_n_steps=None,
        enable_checkpointing=None,
        enable_progress_bar=True,
        enable_model_summary=True,
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        gradient_clip_val=config['training']['clip_grad']['value'],
        gradient_clip_algorithm=config['training']['clip_grad']['norm'],
        deterministic=config['training']['deterministic'],
        benchmark=None,
        use_distributed_sampler=True,
        profiler=None,
        detect_anomaly=False,
        barebones=False,
        plugins=None,
        sync_batchnorm=False,
        reload_dataloaders_every_n_epochs=0,
        default_root_dir=config['project']['results_path'],
    )
    return trainer
