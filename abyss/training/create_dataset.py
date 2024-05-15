import json
import os

import torch.distributed as dist
from monai.data import CacheDataset, DataLoader, partition_dataset

from .transforms import get_transforms


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

        dataset = CacheDataset(
            data=datalist,
            transform=transform,
            num_workers=config['training']['num_workers'],
            cache_rate=config['training']['cache_rate'],
            copy_cache=False,
        )

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config['training']['num_workers'],
            drop_last=False,
        )

    if mode == 'train':
        if config['training']['multi_gpu']:
            datalist = partition_dataset(
                data=datalist,
                shuffle=True,
                num_partitions=dist.get_world_size(),
                even_divisible=True,
            )[dist.get_rank()]

        dataset = CacheDataset(
            data=datalist,
            transform=transform,
            num_workers=config['training']['num_workers'],
            cache_rate=config['training']['cache_rate'],
            copy_cache=False,
        )

        return DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            drop_last=True,
        )

    raise ValueError('mode should be train, validation or test.')
