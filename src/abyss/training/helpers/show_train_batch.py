# pylint: disable-all

import os
import typing as t

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from abyss.config import ConfigManager
from abyss.training.augmentation.augmentation import transforms
from abyss.training.dataset import Dataset


def show_train_batch(save_path: t.Optional[str] = None):
    """Visualize ce train batch"""

    config_manager = ConfigManager()
    config_manager()
    params = config_manager.params
    path_memory = config_manager.path_memory
    ori_dataset = Dataset(params, path_memory, 'train')
    aug_dataset = Dataset(params, path_memory, 'train', transforms)

    ori_loader = DataLoader(ori_dataset, 1)
    aug_loader = DataLoader(aug_dataset, 1)

    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    axes = plt.gca()
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)

    idx = 0
    while True:
        for (ori_data, ori_label), (aug_data, aug_label) in zip(ori_loader, aug_loader):

            plt.subplot(1, 2, 1)
            plt.title('Original')
            plt.imshow(ori_data[0, 0], cmap='gray')
            plt.imshow(ori_label[0, 0], cmap='gray', alpha=0.5)

            plt.subplot(1, 2, 2)
            plt.title('Augmented')
            plt.imshow(aug_data[0, 0], cmap='gray')
            plt.imshow(aug_label[0, 0], cmap='gray', alpha=0.5)

            if save_path:
                idx += 1
                plt.savefig(os.path.join(save_path, f'loaded_batch_{idx}.png'))
            else:
                plt.draw()
                plt.waitforbuttonpress(0)


if __name__ == '__main__':
    show_train_batch()
