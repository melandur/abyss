import json
import os

import torch.distributed as dist
from monai.data import CacheDataset, DataLoader, partition_dataset

from abyss.training.transforms import get_segmentation_transforms


def get_loader(config: dict, mode: str) -> DataLoader:
    """Get the dataloader for training, validation or test."""

    transform = get_segmentation_transforms(config, mode)

    train_dataset_path = config['project']['train_dataset_path']
    train_dataset_file = os.path.join(config['project']['config_path'], 'train_dataset.json')

    with open(train_dataset_file, 'r') as path:
        data_dict = json.load(path)

    if mode == 'test':
        datalist = data_dict['test']
    else:
        datalist = data_dict[f'{mode}_fold_{config["training"]["fold"]}']

    for subject in datalist:
        subject['image'] = [os.path.join(train_dataset_path, subject['name'], image) for image in subject['image']]
        subject['label'] = [os.path.join(train_dataset_path, subject['name'], label) for label in subject['label']]

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
            batch_size=2,
            shuffle=True,
            num_workers=config['training']['num_workers'],
            drop_last=True,
        )

    raise ValueError('mode should be train, validation or test.')


if __name__ == '__main__':
    from config import ConfigFile

    config = ConfigFile().get_config()
    get_loader(config, 'train')
