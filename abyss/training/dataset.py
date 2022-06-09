import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset as torch_Dataset


class Dataset(torch_Dataset):
    """Can be used to create dataset for train, val & test set"""

    def __init__(self, config_manager, set_name, transforms=None):
        super().__init__()
        self.config_manager = config_manager
        self.set_name = set_name
        self.transforms = transforms
        self.params = config_manager.params
        self.dataset_paths = config_manager.get_path_memory(f'{set_name}_dataset_paths')
        set_case_names = list(self.dataset_paths['data'].keys())
        random.seed(self.config_manager.params['meta']['seed'])
        self.random_set_case_names = random.sample(set_case_names, len(set_case_names))
        h5_file_path = os.path.join(self.params['project']['trainset_store_path'], 'data.h5')
        self.h5_object = h5py.File(h5_file_path, 'r')

    def __del__(self):
        if isinstance(self.h5_object, h5py.File):
            self.h5_object.close()

    def concatenate_data(self, case_name):
        """Load from hdf5 and stack data on new first dimensions"""
        img = None
        for idx, file_tag in enumerate(self.dataset_paths['data'][case_name]):
            tmp_img = self.h5_object.get(f'{self.set_name}/data/{case_name}/{file_tag}')
            tmp_img = np.asarray(tmp_img)
            if self.transforms:
                tmp_img = self.transforms(tmp_img)
            if idx == 0:
                img = np.expand_dims(tmp_img, axis=0)
            else:
                tmp_img = np.expand_dims(tmp_img, axis=0)
                img = np.concatenate((img, tmp_img), axis=0)
        return torch.from_numpy(img)

    def retrieve_label(self, case_name):
        """Load label from hdf5"""
        if len(self.dataset_paths['label'][case_name]) > 1:
            raise NotImplementedError('Only 1 label tag supported, adjust this method to your needs')
        for file_tag in self.dataset_paths['label'][case_name]:
            label = self.h5_object.get(f'{self.set_name}/label/{case_name}/{file_tag}')
            return torch.from_numpy(np.asarray(label))

    def __getitem__(self, index):
        case_name = self.random_set_case_names[index]
        data = self.concatenate_data(case_name)
        label = self.retrieve_label(case_name)
        return data, label

    def __len__(self):
        return len(self.random_set_case_names)
