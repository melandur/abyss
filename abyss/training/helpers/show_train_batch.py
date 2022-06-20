import os
from typing import Optional

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from abyss.config import ConfigManager
from abyss.training.augmentation.augmentation import data_transforms
from abyss.training.dataset import Dataset


def show_train_batch(save_path: Optional[str] = None):
    """Visualize ce train batch"""

    cm = ConfigManager()
    cm()
    params = cm.params
    path_memory = cm.path_memory
    ori_dataset = Dataset(params, path_memory, 'train')
    aug_dataset = Dataset(params, path_memory, 'train', data_transforms)

    ori_loader = DataLoader(ori_dataset, 1)
    aug_loader = DataLoader(aug_dataset, 1)

    slice_numbers = [60, 70, 80]
    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    idx = 0
    while True:
        for (ori_data, ori_label), (aug_data, aug_label) in zip(ori_loader, aug_loader):
            for slice_number in slice_numbers:
                for modality in range(len(ori_data[0])):
                    plt.subplot(4, 4, modality + 1)
                    plt.title('Original')
                    plt.imshow(ori_data[0, modality, slice_number], cmap='gray')
                    plt.subplot(4, 4, modality + 5)
                    plt.imshow(ori_label[0, 0, slice_number], cmap='gray')
                    plt.subplot(4, 4, modality + 9)
                    plt.title('Augmented')
                    plt.imshow(aug_data[0, modality, slice_number], cmap='gray')
                    plt.subplot(4, 4, modality + 13)
                    plt.imshow(aug_label[0, 1, slice_number], cmap='gray')
                plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
                idx += 1
                if save_path:
                    print(f'saved image_{idx}_{slice_number}.png')
                    plt.savefig(os.path.join(save_path, f'image_{idx}_{slice_number}.png'))
                else:
                    plt.draw()
                    plt.waitforbuttonpress(0)


if __name__ == '__main__':
    show_train_batch()
    # show_train_batch('/home/melandur/Downloads/aby/2')
