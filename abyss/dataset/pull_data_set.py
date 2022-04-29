import os

from loguru import logger
from monai.apps import CrossValidation, DecathlonDataset, MedNISTDataset


def pull_data_set(_params):
    """Allows to download data from API"""
    os.makedirs(_params['project']['dataset_store_path'], exist_ok=True)
    dataset = _params['dataset']['pull_dataset']

    if 'MedNISTDataset' in dataset:
        logger.info(f'using: {dataset}')
        MedNISTDataset(
            root_dir=_params['project']['dataset_store_path'],
            section='training',
            download=True,
            seed=_params['dataset']['seed'],
            val_frac=_params['dataset']['val_frac'],
            test_frac=_params['dataset']['test_frac'],
            num_workers=_params['dataset']['num_workers'],
        )

    elif 'DecathlonDataset' in dataset:
        logger.info(f'using: {dataset}, {_params["dataset"]["challenge"]}')
        DecathlonDataset(
            root_dir=_params['project']['dataset_store_path'],
            task=_params['dataset']['challenge'],
            section='training',
            download=True,
            seed=_params['dataset']['seed'],
            val_frac=_params['dataset']['val_frac'],
            cache_num=_params['dataset']['cache_max'],
            cache_rate=_params['dataset']['cache_rate'],
            num_workers=_params['dataset']['num_workers'],
        )

        DecathlonDataset(
            root_dir=_params['project']['dataset_store_path'],
            task=_params['dataset']['challenge'],
            section='validation',
            download=False,
            num_workers=_params['dataset']['num_workers'],
        )

        DecathlonDataset(
            transform=(),
            root_dir=_params['project']['dataset_store_path'],
            task=_params['dataset']['challenge'],
            section='test',
            download=True,
            cache_num=_params['dataset']['cache_max'],
            cache_rate=_params['dataset']['cache_rate'],
            num_workers=_params['dataset']['num_workers'],
        )

    elif 'CrossValidation' in dataset:
        logger.info(f'using: {dataset}')
        CrossValidation(
            dataset_cls=None,
            root_dir=_params['project']['dataset_store_path'],
            task=_params['dataset']['challenge'],
            section='training',
            download=True,
            num_workers=_params['dataset']['num_workers'],
            cache_num=_params['dataset']['cache_num'],
        )

    # elif 'CustomDataset' in _params['dataset']['dataset']:
    #     print(f'using: {dataset}')
    #     TODO: Need some code, split data into imageTr, imageTs, labelTr, labelTs. Stored as nii.gz

    else:
        raise NotImplementedError(
            "Invalid dataset settings in conf.py: _params['data']['dataset'], "
            "options: CustomDataset, DecathlonDataset, MedNISTDataset, CrossValidation"
        )


if __name__ == '__main__':
    from abyss.config.config_manager import ConfigManager

    params = ConfigManager()
    pull_data_set(params)
