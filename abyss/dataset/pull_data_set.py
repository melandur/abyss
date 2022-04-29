import os

from loguru import logger
from monai.apps import CrossValidation, DecathlonDataset, MedNISTDataset


def pull_data_set(_params):
    """Allows to download data from API"""
    os.makedirs(_params['project']['dataset_folder_path'], exist_ok=True)
    dataset = _params['dataset']['pull_dataset']['dataset_name']

    if 'MedNISTDataset' in dataset:
        logger.info(f'pull: {dataset}')
        MedNISTDataset(
            root_dir=_params['project']['dataset_folder_path'],
            section='training',
            download=True,
            seed=_params['dataset']['pull_dataset']['seed'],
            val_frac=_params['dataset']['pull_dataset']['val_frac'],
            test_frac=_params['dataset']['pull_dataset']['test_frac'],
            num_workers=_params['dataset']['pull_dataset']['num_workers'],
        )

    elif 'DecathlonDataset' in dataset:
        logger.info(f'pull: {dataset}, {_params["dataset"]["pull_dataset"]["challenge"]}')
        DecathlonDataset(
            root_dir=_params['project']['dataset_folder_path'],
            task=_params['dataset']['pull_dataset']['challenge'],
            section='training',
            download=True,
            seed=_params['dataset']['pull_dataset']['seed'],
            val_frac=_params['dataset']['pull_dataset']['val_frac'],
            cache_num=_params['dataset']['pull_dataset']['cache_max'],
            cache_rate=_params['dataset']['pull_dataset']['cache_rate'],
            num_workers=_params['dataset']['pull_dataset']['num_workers'],
        )

        DecathlonDataset(
            root_dir=_params['project']['dataset_folder_path'],
            task=_params['dataset']['pull_dataset']['challenge'],
            section='validation',
            download=False,
            num_workers=_params['dataset']['pull_dataset']['num_workers'],
        )

        DecathlonDataset(
            transform=(),
            root_dir=_params['project']['dataset_folder_path'],
            task=_params['dataset']['pull_dataset']['challenge'],
            section='test',
            download=True,
            cache_num=_params['dataset']['pull_dataset']['cache_max'],
            cache_rate=_params['dataset']['pull_dataset']['cache_rate'],
            num_workers=_params['dataset']['pull_dataset']['num_workers'],
        )

    elif 'CrossValidation' in dataset:
        logger.info(f'pull: {dataset}')
        CrossValidation(
            dataset_cls=None,
            root_dir=_params['project']['dataset_folder_path'],
            task=_params['dataset']['pull_dataset']['challenge'],
            section='training',
            download=True,
            num_workers=_params['dataset']['pull_dataset']['num_workers'],
            cache_num=_params['dataset']['pull_dataset']['cache_num'],
        )
    else:
        raise NotImplementedError(
            "Invalid dataset settings in conf.py: _params['data']['dataset'], "
            "options: CustomDataset, DecathlonDataset, MedNISTDataset, CrossValidation"
        )
