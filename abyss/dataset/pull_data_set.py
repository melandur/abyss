import os

from loguru import logger as log
from monai.apps import CrossValidation, DecathlonDataset, MedNISTDataset


def pull_data_set(params):
    """Allows to download data from API"""
    os.makedirs(params['project']['dataset_store_path'], exist_ok=True)
    dataset = params['dataset']['pull_dataset']

    if 'MedNISTDataset' in dataset:
        log.info(f'using: {dataset}')
        train_ds = MedNISTDataset(
            root_dir=params['project']['dataset_store_path'],
            section='training',
            download=True,
            seed=params['dataset']['seed'],
            val_frac=params['dataset']['val_frac'],
            test_frac=params['dataset']['test_frac'],
            num_workers=params['dataset']['num_workers'],
        )

    elif 'DecathlonDataset' in dataset:
        log.info(f'using: {dataset}, {params["dataset"]["challenge"]}')
        DecathlonDataset(
            root_dir=params['project']['dataset_store_path'],
            task=params['dataset']['challenge'],
            section='training',
            download=True,
            seed=params['dataset']['seed'],
            val_frac=params['dataset']['val_frac'],
            cache_num=params['dataset']['cache_max'],
            cache_rate=params['dataset']['cache_rate'],
            num_workers=params['dataset']['num_workers'],
        )

        DecathlonDataset(
            root_dir=params['project']['dataset_store_path'],
            task=params['dataset']['challenge'],
            section='validation',
            download=False,
            num_workers=params['dataset']['num_workers'],
        )

        DecathlonDataset(
            transform=(),
            root_dir=params['project']['dataset_store_path'],
            task=params['dataset']['challenge'],
            section='test',
            download=True,
            cache_num=params['dataset']['cache_max'],
            cache_rate=params['dataset']['cache_rate'],
            num_workers=params['dataset']['num_workers'],
        )

    elif 'CrossValidation' in dataset:
        log.info(f'using: {dataset}')
        CrossValidation(
            root_dir=params['project']['dataset_store_path'],
            task=params['dataset']['challenge'],
            section='training',
            download=True,
            num_workers=params['dataset']['num_workers'],
            cache_num=params['dataset']['cache_num'],
        )

    # elif 'CustomDataset' in params['dataset']['dataset']:
    #     print(f'using: {dataset}')
    #     TODO: Need some code, split data into imageTr, imageTs, labelTr, labelTs. Stored as nii.gz

    else:
        raise AssertionError("Invalid dataset settings in conf.py: params['data']['dataset'], "
                             "options: CustomDataset, DecathlonDataset, MedNISTDataset, CrossValidation")


if __name__ == '__main__':
    from src.config.config_manager import ConfigManager

    params = ConfigManager().params
    pull_data_set(params)
