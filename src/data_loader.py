import os
from monai.apps import MedNISTDataset, DecathlonDataset, CrossValidation

from main_conf import params


def get_data_set():
    os.makedirs(params['project']['dataset_store_path'], exist_ok=True)
    dataset = params['data']['dataset']

    if 'MedNISTDataset' in dataset:
        print(f'using: {dataset}')
        train_ds = MedNISTDataset(
            root_dir=params['project']['dataset_store_path'],
            section='training',
            download=True,
            seed=params['data']['seed'],
            val_frac=params['data']['val_frac'],
            test_frac=params['data']['test_frac'],
            num_workers=params['data']['num_workers']
        )

    elif 'DecathlonDataset' in dataset:
        print(f'using: {dataset}, {params["data"]["challenge"]}')

        DecathlonDataset(
            root_dir=params['project']['dataset_store_path'],
            task=params['data']['challenge'],
            section='training',
            download=True,
            seed=params['data']['seed'],
            val_frac=params['data']['val_frac'],
            cache_num=params['data']['cache_max'],
            cache_rate=params['data']['cache_rate'],
            num_workers=params['data']['num_workers']
        )

        DecathlonDataset(
            root_dir=params['project']['dataset_store_path'],
            task=params['data']['challenge'],
            section='validation',
            download=False,
            num_workers=params['data']['num_workers']
        )

        DecathlonDataset(
            transform=(),
            root_dir=params['project']['dataset_store_path'],
            task=params['data']['challenge'],
            section='test',
            download=True,
            cache_num=params['data']['cache_max'],
            cache_rate=params['data']['cache_rate'],
            num_workers=params['data']['num_workers']
        )

    elif 'CrossValidation' in dataset:
        print(f'using: {dataset}')
        CrossValidation(
            root_dir=params['project']['dataset_store_path'],
            task=params['data']['challenge'],
            section='training',
            download=True,
            num_workers=params['data']['num_workers'],
            cache_num=params['data']['cache_num']
        )

    elif 'CustomDataset' in params['data']['dataset']:
        print(f'using: {dataset}')
        # TODO: Need some code, split data into imageTr, imageTs, labelTr, labelTs. Stored as nii.gz

    else:
        print("Invalid dataset settings in conf.py: params['data']['dataset']")
        exit(1)


if __name__ == '__main__':
    print(1)
