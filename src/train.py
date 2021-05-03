import os
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from conf import params
from src.net import Net


if __name__ == '__main__':

    # # initialise the LightningModule
    net = Net()

    # set up loggers and checkpoints
    log_dir = os.path.join(params['user']['dataset_store_path'], 'logs')
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=log_dir)
    checkpoint_callback = ModelCheckpoint(dirpath=log_dir, filename='test')

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(gpus=[0],
                                        max_epochs=params['training']['epochs'],
                                        logger=tb_logger,
                                        checkpoint_callback=True,
                                        num_sanity_val_steps=1)
    # train
    trainer.fit(net)
