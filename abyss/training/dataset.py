import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset as torch_Dataset


class Dataset(torch_Dataset):
    """Can be used to create dataset for train, val & test set"""

    def __init__(self, params: dict, path_memory: dict, set_name: str, transforms=None):
        self.set_name = set_name
        self.transforms = transforms
        self.h5_object = None
        self.dataset_paths = path_memory[f'{set_name}_dataset_paths']
        set_case_names = list(self.dataset_paths['data'].keys())
        random.seed(params['meta']['seed'])
        self.random_set_case_names = random.sample(set_case_names, len(set_case_names))
        h5_file_path = os.path.join(params['project']['trainset_store_path'], 'data.h5')
        if os.path.isfile(h5_file_path):
            self.h5_object = h5py.File(h5_file_path, 'r')
        else:
            raise FileExistsError(f'HDF5 file is missing -> {h5_file_path}')

    def __del__(self):
        """Close hdf5 file in the end"""
        if isinstance(self.h5_object, h5py.File):
            self.h5_object.close()

    def concatenate_data(self, case_name: str) -> torch.tensor:
        """Load from hdf5 and stack data on new first dimensions"""
        img = None
        for idx, file_tag in enumerate(self.dataset_paths['data'][case_name]):
            tmp_img = self.h5_object.get(f'{self.set_name}/data/{case_name}/{file_tag}')
            tmp_img = np.asarray(tmp_img)
            tmp_img = np.expand_dims(tmp_img, axis=0)
            if idx == 0:
                img = tmp_img
            else:
                img = np.concatenate((img, tmp_img), axis=0)
        if self.transforms:
            img = self.transforms(img)
        return torch.from_numpy(img)

    def retrieve_label(self, case_name: str) -> torch.tensor:
        """Load label from hdf5"""
        if len(self.dataset_paths['label'][case_name]) > 1:
            raise NotImplementedError('Only 1 label tag supported, adjust this method to your needs')
        for file_tag in self.dataset_paths['label'][case_name]:
            label = self.h5_object.get(f'{self.set_name}/label/{case_name}/{file_tag}')
            return torch.from_numpy(np.asarray(label))

    def __getitem__(self, index) -> tuple:
        """Returns data and corresponding label"""
        case_name = self.random_set_case_names[index]
        data = self.concatenate_data(case_name)
        label = self.retrieve_label(case_name)
        return data, label

    def __len__(self) -> int:
        """Holds number of cases"""
        return len(self.random_set_case_names)
