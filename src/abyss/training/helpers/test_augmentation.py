# pylint: disable-all

import os

import matplotlib.pyplot as plt
import SimpleITK as sitk
from torch.utils.data import DataLoader

from abyss.config import ConfigManager
from abyss.training.augmentation.augmentation import transforms
from abyss.training.dataset import Dataset


def test_data_augmentation():
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
            # plt.subplot(1, 2, 1)
            # plt.title('Original')
            # plt.imshow(ori_data[0, 0, 50], cmap='gray')
            # plt.imshow(ori_label[0, 0, 50], cmap='gray', alpha=0.5)
            #
            # plt.subplot(1, 2, 2)
            # plt.title('Augmented')
            # plt.imshow(aug_data[0, 0, 50], cmap='gray')
            # plt.imshow(aug_label[0, 0, 50], cmap='gray', alpha=0.5)

            # print(ori_data.shape, ori_label.shape, aug_data.shape, aug_label.shape)
            #
            # save_path = os.path.join(params['project']['result_store_path'], 'test_aug', str(idx))
            # os.makedirs(save_path, exist_ok=True)
            # sitk.WriteImage(sitk.GetImageFromArray(ori_data[0, 0, :, :, :]), os.path.join(save_path,
            #                                                                               f'ori_img_{idx}.nii.gz'))
            # sitk.WriteImage(sitk.GetImageFromArray(ori_label[0, 0, :, :, :]), os.path.join(save_path, f'ori_label_{idx}.nii.gz'))
            # sitk.WriteImage(sitk.GetImageFromArray(aug_data[0, 0, :, :, :]), os.path.join(save_path, f'aug_img_{idx}.nii.gz'))
            # sitk.WriteImage(sitk.GetImageFromArray(aug_label[0, 0, :, :, :]), os.path.join(save_path, f'aug_label_{idx}.nii.gz'))
            idx += 1
            print(idx)


if __name__ == '__main__':
    test_data_augmentation()
