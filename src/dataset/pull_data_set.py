import os
from monai.apps import MedNISTDataset, DecathlonDataset, CrossValidation

"""Allows to download data by API"""

def pull_data_set(params):
    os.makedirs(params['project']['dataset_store_path'], exist_ok=True)
    dataset = params['dataset']['pull_dataset']

    if 'MedNISTDataset' in dataset:
        print(f'using: {dataset}')
        train_ds = MedNISTDataset(
            root_dir=params['project']['dataset_store_path'],
            section='training',
            download=True,
            seed=params['dataset']['seed'],
            val_frac=params['dataset']['val_frac'],
            test_frac=params['dataset']['test_frac'],
            num_workers=params['dataset']['num_workers']
        )

    elif 'DecathlonDataset' in dataset:
        print(f'using: {dataset}, {params["dataset"]["challenge"]}')

        DecathlonDataset(
            root_dir=params['project']['dataset_store_path'],
            task=params['dataset']['challenge'],
            section='training',
            download=True,
            seed=params['dataset']['seed'],
            val_frac=params['dataset']['val_frac'],
            cache_num=params['dataset']['cache_max'],
            cache_rate=params['dataset']['cache_rate'],
            num_workers=params['dataset']['num_workers']
        )

        DecathlonDataset(
            root_dir=params['project']['dataset_store_path'],
            task=params['dataset']['challenge'],
            section='validation',
            download=False,
            num_workers=params['dataset']['num_workers']
        )

        DecathlonDataset(
            transform=(),
            root_dir=params['project']['dataset_store_path'],
            task=params['dataset']['challenge'],
            section='test',
            download=True,
            cache_num=params['dataset']['cache_max'],
            cache_rate=params['dataset']['cache_rate'],
            num_workers=params['dataset']['num_workers']
        )

    elif 'CrossValidation' in dataset:
        print(f'using: {dataset}')
        CrossValidation(
            root_dir=params['project']['dataset_store_path'],
            task=params['dataset']['challenge'],
            section='training',
            download=True,
            num_workers=params['dataset']['num_workers'],
            cache_num=params['dataset']['cache_num']
        )

    # elif 'CustomDataset' in params['dataset']['dataset']:
    #     print(f'using: {dataset}')
    #     TODO: Need some code, split data into imageTr, imageTs, labelTr, labelTs. Stored as nii.gz

    # else:
        # print("Invalid dataset settings in conf.py: params['data']['dataset']")
        # exit(1)

if __name__ == '__main__':
    from src.config.config_manager import ConfigManager
    params = ConfigManager().params
    pull_data_set(params)