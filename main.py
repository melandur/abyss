import os
import resource
import warnings

from loguru import logger

# Suppress deprecation warning from fft-conv-pytorch (third-party library issue)
# This warning is from fft_conv_pytorch using deprecated indexing that will break in PyTorch 2.9
warnings.filterwarnings(
    'ignore',
    message='Using a non-tuple sequence for multidimensional indexing is deprecated',
    category=UserWarning,
    module='fft_conv_pytorch',
)

# Suppress informational warning from PyTorch Lightning model summary
# Model size estimation is not accurate with 16-mixed precision, but this doesn't affect training
warnings.filterwarnings(
    'ignore',
    message='Precision 16-mixed is not supported by the model summary',
    category=UserWarning,
    module='pytorch_lightning',
)

from abyss.config import ConfigFile
from abyss.data.create_datalist import create_datalist, create_train_dataset_file
from abyss.engine.trainer import get_trainer
from abyss.models.model import Model
from abyss.utils.dataset_fingerprint import fingerprint_dataset

# Increase the number of file descriptors to the maximum allowed
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

config_file = ConfigFile()
config = config_file.get_config()

# Create dataset files if they don't exist (train_dataset.json and folds)
if config['mode']['train']:
    train_dataset_file = os.path.join(config['project']['config_path'], 'train_dataset.json')

    if not os.path.exists(train_dataset_file):
        logger.info('Creating dataset files (train_dataset.json and folds)')
        create_train_dataset_file(config)
        create_datalist(config)
        logger.success('Dataset files created successfully')
    else:
        # Check if folds exist in the file
        import json

        with open(train_dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # Check if folds are missing (only 'training' key exists)
        if 'training' in dataset and not any(key.startswith('train_fold_') for key in dataset.keys()):
            logger.info('Creating dataset folds')
            create_datalist(config)
            logger.success('Dataset folds created successfully')

    # Run dataset fingerprinting to compute automatic parameters (nnUNet v2 style)
    logger.info('Running dataset fingerprinting')
    config = fingerprint_dataset(config)

model = Model(config)
trainer = get_trainer(config)


if config['mode']['train']:
    ckpt_path = None
    if config['training']['reload_checkpoint']:
        if os.path.exists(config['project']['results_path']):
            ckpt_path = os.path.join(config['project']['results_path'], 'last.ckpt')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f'No last found in -> {config["project"]["results_path"]}')
    trainer.fit(model, ckpt_path=ckpt_path)


if config['mode']['test']:
    if config['training']['checkpoint_path'] is not None:
        if not os.path.exists(config['training']['checkpoint_path']):
            raise FileNotFoundError(f'No checkpoint found in -> {config["training"]["checkpoint_path"]}')
        ckpt_path = config['training']['checkpoint_path']
        trainer.test(model, ckpt_path=ckpt_path)
