import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset as torch_Dataset


class Dataset(torch_Dataset):
    """Can be used to create dataset for train, val & test set"""

    def __init__(self, params: dict, path_memory: dict, set_name: str, transforms=None) -> None:
        self.params = params
        self.set_name = set_name
        self.transforms = transforms
        self.dataset_paths = path_memory[f'{set_name}_dataset_paths']
        set_case_names = list(self.dataset_paths)
        random.seed(params['meta']['seed'])
        self.random_set_case_names = random.sample(set_case_names, len(set_case_names))
        h5_file_path = os.path.join(params['project']['trainset_store_path'], 'data.h5')
        self.h5_object = None
        if os.path.isfile(h5_file_path):
            self.h5_object = h5py.File(h5_file_path, 'r')
        else:
            raise FileExistsError(f'HDF5 file is missing -> {h5_file_path}')

    def __del__(self) -> None:
        """Close hdf5 file in the end"""
        if isinstance(self.h5_object, h5py.File):
            self.h5_object.close()

    def prepare_data(self, case_name: str) -> torch.tensor:
        """Load from hdf5 and stack/add_dimension or what ever"""
        t1c_img = self.h5_object.get(f'{self.set_name}/{case_name}/data/images/t1c')
        t1_img = self.h5_object.get(f'{self.set_name}/{case_name}/data/images/t1')
        t2_img = self.h5_object.get(f'{self.set_name}/{case_name}/data/images/t2')
        flair_img = self.h5_object.get(f'{self.set_name}/{case_name}/data/images/flair')
        t1c_arr = np.asarray(t1c_img, dtype='float32')
        t1_arr = np.asarray(t1_img, dtype='float32')
        t2_arr = np.asarray(t2_img, dtype='float32')
        flair_arr = np.asarray(flair_img, dtype='float32')
        img = np.stack([t1c_arr, t1_arr, t2_arr, flair_arr], axis=0)  # stack channels
        return torch.from_numpy(img)

    def prepare_label(self, case_name: str) -> torch.tensor:
        """Load label from hdf5 and stack/add_dimension or what ever"""
        label = self.h5_object.get(f'{self.set_name}/{case_name}/labels/images/mask')
        label = np.asarray(label, dtype='int8')  # TODO: int it goes
        label = np.expand_dims(label, axis=0)  # add channel
        return torch.from_numpy(label)

    def __getitem__(self, index) -> tuple:
        """Returns data and corresponding label"""
        case_name = self.random_set_case_names[index]
        data = self.prepare_data(case_name)
        label = self.prepare_label(case_name)
        sample = {'data': data, 'labels': label}
        if self.transforms:
            sample = self.transforms(sample)
        return sample['data'], sample['labels']

    def __len__(self) -> int:
        """Holds number of cases"""
        return len(self.random_set_case_names)
