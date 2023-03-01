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

    def __del__(self) -> None:
        """Close hdf5 file in the end"""
        if isinstance(self.h5_object, h5py.File):
            self.h5_object.close()

    def prepare_data(self, case_name: str, slice_index: int) -> torch.tensor:
        """Load from hdf5 and stack/add_dimension or what ever"""
        t1 = self.h5_object.get(f'{self.set_name}/data/{case_name}/t1')
        t1c = self.h5_object.get(f'{self.set_name}/data/{case_name}/t1c')
        t2 = self.h5_object.get(f'{self.set_name}/data/{case_name}/t2')
        flair = self.h5_object.get(f'{self.set_name}/data/{case_name}/flair')
        img = np.concatenate([t1, t1c, t2, flair])
        if img is None:
            raise FileNotFoundError('Image not found, this method needs to be adjusted to your needs')
        return torch.from_numpy(img).type(torch.FloatTensor)

    def prepare_label(self, case_name: str, slice_index: int) -> torch.tensor:
        """Load label from hdf5 and stack/add_dimension or what ever"""
        if len(self.dataset_paths['label'][case_name]) > 1:
            raise NotImplementedError('Only 1 label tag is supported, adjust this method to your needs')
        label = self.h5_object.get(f'{self.set_name}/label/{case_name}/mask')
        # label = np.squeeze(label, axis=0)
        label = np.asarray(label)
        return torch.from_numpy(label).type(torch.LongTensor)

    def __getitem__(self, index) -> tuple:
        """Returns data and corresponding label"""
        slices = 128  # todo: get from params
        case_index, slice_index = divmod(index, slices)
        case_name = self.random_set_case_names[case_index]

        data = self.prepare_data(case_name, slice_index)
        label = self.prepare_label(case_name, slice_index)
        sample = {'data': data, 'label': label}
        if self.transforms:
            sample = self.transforms(sample)
        sample['label'] = torch.squeeze(sample['label'], dim=0).type(
            torch.LongTensor
        )  # todo: maybe monai can do this in augmentation
        return sample['data'], sample['label']

    def __len__(self) -> int:
        """Holds number of cases"""
        return len(self.random_set_case_names)
