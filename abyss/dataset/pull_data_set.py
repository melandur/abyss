import sys
import os

from loguru import logger
from monai.apps import DecathlonDataset, MedNISTDataset


def pull_dataset(_params, _dataset_folder_path):
    """Allows to download data from API"""
    os.makedirs(_dataset_folder_path, exist_ok=True)
    dataset = _params['dataset_name']

    if 'MedNISTDataset' in dataset:
        logger.info(f'pull: {dataset}')
        MedNISTDataset(
            root_dir=_dataset_folder_path,
            section='training',
            download=True,
            seed=_params['seed'],
            val_frac=_params['val_frac'],
            test_frac=_params['test_frac'],
            num_workers=_params['num_workers'],
        )

    elif 'DecathlonDataset' in dataset:
        logger.info(f'pull: {dataset}, {_params["challenge"]}')
        DecathlonDataset(
            root_dir=_dataset_folder_path,
            task=_params['challenge'],
            section='training',
            download=True,
            seed=_params['seed'],
            val_frac=_params['val_frac'],
            cache_num=_params['cache_max'],
            cache_rate=_params['cache_rate'],
            num_workers=_params['num_workers'],
        )

        DecathlonDataset(
            root_dir=_params['project']['dataset_folder_path'],
            task=_params['challenge'],
            section='validation',
            download=False,
            num_workers=_params['num_workers'],
        )

        DecathlonDataset(
            transform=(),
            root_dir=_params['project']['dataset_folder_path'],
            task=_params['challenge'],
            section='test',
            download=True,
            cache_num=_params['cache_max'],
            cache_rate=_params['cache_rate'],
            num_workers=_params['num_workers'],
        )
    else:
        raise NotImplementedError(
            "Invalid dataset settings in conf.py: _params['data']['dataset'], "
            "options: CustomDataset, DecathlonDataset, MedNISTDataset, CrossValidation"
        )


if __name__ == '__main__':

    dataset_folder_path = os.path.join(os.path.expanduser('~'), 'Downloads')
    params = {
        'dataset_name': 'MedNISTDataset',  # 'MedNISTDataset', 'DecathlonDataset'
        'challenge': 'Task01_BrainTumour',  # only need for decathlon:   'Task01_BrainTumour',
        # 'Task02_Heart', 'Task03_Liver0', 'Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung',
        # 'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon'
        'seed': 42,  # find the truth in randomness
        'val_frac': 0.2,
        'test_frac': 0.2,
        'use_cache': False,  # if true goes heavy on memory
        'cache_max': sys.maxsize,
        'cache_rate': 0.0,  # 0.0 minimal memory footprint, 1.0 goes heavy on memory
        'num_workers': 8,
    }

    pull_dataset(params, dataset_folder_path)
