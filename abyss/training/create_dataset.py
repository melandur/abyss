import json
import os

import torch.distributed as dist
from monai.data import CacheDataset, DataLoader, partition_dataset
from transforms import get_transforms


def get_loader(config: dict, mode: str):
    """Get the dataloader for training, validation or test."""
    transform = get_transforms(config, mode)

    dataset_file_path = os.path.join(config['project']['config_path'], 'dataset.json')
    with open(dataset_file_path, 'r') as path:
        data_dict = json.load(path)

    datalist = data_dict[f'{mode}_fold_{config["training"]["fold"]}']

    if mode in ['val', 'test']:
        if config['training']['multi_gpu']:
            datalist = partition_dataset(
                data=datalist,
                shuffle=False,
                num_partitions=dist.get_world_size(),
                even_divisible=False,
            )[dist.get_rank()]

        val_ds = CacheDataset(
            data=datalist,
            transform=transform,
            num_workers=config['training']['num_workers']['val'],
            cache_rate=config['training']['cache_rate'],
        )

        data_loader = DataLoader(
            val_ds,
            batch_size=config['training']['batch_size']['val'],
            shuffle=False,
            num_workers=config['training']['num_workers']['val'],
            drop_last=False,
        )
        return data_loader

    if mode == 'train':
        if config['training']['multi_gpu']:
            datalist = partition_dataset(
                data=datalist,
                shuffle=True,
                num_partitions=dist.get_world_size(),
                even_divisible=True,
            )[dist.get_rank()]

        train_ds = CacheDataset(
            data=datalist,
            transform=transform,
            num_workers=config['training']['num_workers']['train'],
            cache_rate=config['training']['cache_rate'],
        )
        data_loader = DataLoader(
            train_ds,
            batch_size=config['training']['batch_size']['train'],
            shuffle=True,
            num_workers=config['training']['num_workers']['train'],
            drop_last=True,
        )
        return data_loader

    raise ValueError(f'mode should be train, validation or test.')
