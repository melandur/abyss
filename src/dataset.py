from monai.apps import MedNISTDataset, DecathlonDataset, CrossValidation

from conf import params
from src.transform import train_transform, val_transform

dataset = params['data']['dataset']
if 'MedNISTDataset' in dataset:
    print(f'using: {dataset}')
    train_ds = MedNISTDataset(
        root_dir=params['user']['dataset_store_path'],
        section='training',
        transform=train_transform,
        download=True,
        seed=params['data']['seed'],
        val_frac=params['data']['val_frac'],
        test_frac=params['data']['test_frac'],
        num_workers=params['data']['num_workers']
    )
    # val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    # test_ds = MedNISTDataset(test_x, test_y, val_transforms)


elif 'DecathlonDataset' in dataset:
    print(f'using: {dataset}')
    train_ds = DecathlonDataset(
        root_dir=params['user']['dataset_store_path'],
        task=params['data']['challenge'],
        transform=train_transform,
        section='training',
        download=True,
        num_workers=params['data']['num_workers'],
        val_frac=params['data']['val_frac'],
        seed=params['data']['seed']
    )

    val_ds = DecathlonDataset(
        root_dir=params['user']['dataset_store_path'],
        task=params['data']['challenge'],
        transform=val_transform,
        section='validation',
        download=False,
        num_workers=params['data']['num_workers']
    )

elif 'CrossValidation' in dataset:
    print(f'using: {dataset}')
    train_ds = CrossValidation(
        root_dir=params['user']['dataset_store_path'],
        task=params['data']['challenge'],
        transform=train_transform,
        section='training',
        download=True,
        num_workers=params['data']['num_workers'],
        cache_num=params['data']['cache_num']
    )

elif 'CustomDataset' in params['data']['dataset']:
    # TODO: Need some code
    pass

else:
    print("Invalid dataset settings in conf.py: params['data']['dataset']")
    exit(1)


