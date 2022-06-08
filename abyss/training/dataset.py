import os
import h5py
import torch
from torch.utils.data import Dataset as torch_Dataset


class Dataset(torch_Dataset):
    def __init__(self, _config_manager):
        super().__init__()
        self.config_manager = _config_manager
        self.params = _config_manager.params

        self.train_set_cases = _config_manager.get_path_memory('train_dataset_paths')
        self.val_set_cases = None
        self.test_set_cases = None

        print(self.train_set_cases)

        # self.h5_file_path = os.path.join(self.params['project']['trainset_store_path'])
        # if not os.path.isfile:
        #     raise ValueError(f'HDF5 file not found -> {self.h5_file_path}')


        # self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        # self.indices = {}
        # idx = 0
        # for a, archive in enumerate(self.archives):
        #     for i in range(len(archive)):
        #         self.indices[idx] = (a, i)

    # @property
    # def archives(self):
    #     if self._archives is None:  # lazy loading here!
    #         self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
    #     return self._archives

    # def __getitem__(self, index):
    #     a, i = self.indices[index]
    #     archive = self.archives[a]
    #     dataset = archive[f"trajectory_{i}"]
    #     data = torch.from_numpy(dataset[:])
    #     labels = dict(dataset.attrs)
    #
    #     return {"data": data, "labels": labels}
    #
    # def __len__(self):
    #     if self.limit > 0:
    #         return min([len(self.indices), self.limit])
    #     return len(self.indices)

if __name__ == '__main__':
    from abyss.config import ConfigManager
    cm = ConfigManager()
    d = Dataset(cm)
    # for x in d:
    #     print(x)