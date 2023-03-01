import copy
import os

import h5py
import numpy as np
import torchio as tio
from loguru import logger

from abyss.config import ConfigManager
from abyss.utils import NestedDefaultDict


class CreateHDF5(ConfigManager):
    """Create train, val, and test set for training"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)
        self.trainset_store_path = os.path.join(self.params['project']['trainset_store_path'], 'data.h5')
        self.train_set_cases = None
        self.val_set_cases = None
        self.test_set_cases = None
        self.use_val_fraction = False
        self.data_store_paths = None
        self.tree_store = ''
        np.random.seed(self.params['meta']['seed'])

    def __call__(self) -> None:
        if self.params['pipeline_steps']['create_trainset']:
            logger.info(f'Run: {self.__class__.__name__}')
            self.data_store_paths = self.get_data_store_paths()
            self.check_fold_settings()
            self.train_test_split()
            self.train_val_split()
            self.contamination_check()
            self.execute_dataset_split()
            self.show_tree_structure()
            self.store_tree_structure()
            self.store_path_memory_file()

    def get_data_store_paths(self) -> NestedDefaultDict:
        """Returns the current data store path, with prio 1: pre processed, prio 2: structured dataset"""
        if len(self.path_memory['pre_processed_dataset_paths']['data']) != 0:
            logger.info('HDF5 file will be created from pre processed dataset')
            return self.path_memory['pre_processed_dataset_paths']
        if len(self.path_memory['structured_dataset_paths']['data']) != 0:
            logger.info('HDF5 file will be created from structured dataset')
            return self.path_memory['structured_dataset_paths']
        raise ValueError(
            'Path memory file is empty for structured and pre processed data, check config_file -> pipeline_steps'
        )

    def train_test_split(self) -> None:
        """Creates a list with case names for train and test set each"""
        count_cases = len(self.data_store_paths['data'])
        if count_cases < 10:
            raise AssertionError('Your dataset needs to have at least 10 subjects')
        test_set_size = int(self.params['dataset']['test_fraction'] * count_cases)
        self.test_set_cases = list(
            np.random.choice(list(self.data_store_paths['data']), size=test_set_size, replace=False)
        )
        self.train_set_cases = [x for x in self.data_store_paths['data'] if x not in self.test_set_cases]
        logger.info(f'Test set, counts: {len(self.test_set_cases)}, cases: {self.test_set_cases}')

    def check_fold_settings(self) -> None:
        """Check for valid fold settings"""
        if '/' not in self.params['dataset']['cross_fold']:
            raise ValueError('config_file -> dataset -> cross_fold: fold_number/max_number_of_folds')
        max_folds = int(self.params['dataset']['cross_fold'].split('/')[1])
        if max_folds <= 0:
            raise ValueError('config_file -> dataset -> cross_fold: max_number_of_folds >= 1')
        if max_folds == 1:
            self.use_val_fraction = True
            logger.info(
                f'Max_number_of_folds = 1, therefore the current val_fraction of '
                f'{self.params["dataset"]["val_fraction"]} will be used to determine the validation set size'
            )
        fold_number = int(self.params['dataset']['cross_fold'].split('/')[0])
        if fold_number <= 0:
            raise ValueError('config_file -> dataset -> cross_fold: fold_number >= 1')
        if fold_number > max_folds:
            raise ValueError('fold_number exceeded max_number_of_folds, check cross_fold settings')

    def train_val_split(self) -> None:
        """Split train data into train and val data, determined by val_fraction or by cross_fold"""
        count_cases = len(self.train_set_cases)
        if self.use_val_fraction:
            val_set_size = int(round(self.params['dataset']['val_fraction'] * count_cases))
        else:
            fold_fraction = round(1.0 / int(self.params['dataset']['cross_fold'].split('/')[1]), 2)
            val_set_size = int(round(fold_fraction * count_cases))
        fold_number = int(self.params['dataset']['cross_fold'].split('/')[0])
        if val_set_size <= 0:
            raise ValueError(
                f'Validation set has {val_set_size} cases, config_file -> increase val_fraction or reduce test_fraction'
            )
        tmp_train_set_cases = copy.deepcopy(self.train_set_cases)
        for _ in range(1, fold_number + 1, 1):
            self.val_set_cases = list(np.random.choice(tmp_train_set_cases, size=val_set_size, replace=False))
        self.train_set_cases = [x for x in self.train_set_cases if x not in self.val_set_cases]
        logger.info(f'Train set, counts: {len(self.train_set_cases)}, cases: {self.train_set_cases}')
        logger.info(f'Val set, counts: {len(self.val_set_cases)}, cases: {self.val_set_cases}')

    def contamination_check(self) -> None:
        """Assures that cases are unique in each dataset"""
        contamination = set(self.test_set_cases) & set(self.train_set_cases)
        if contamination:
            raise AssertionError(f'Contamination in train & test-set split -> {contamination}')
        contamination = set(self.train_set_cases) & set(self.val_set_cases)
        if contamination:
            raise AssertionError(f'Contamination in train & val-set split -> {contamination}')
        contamination = set(self.test_set_cases) & set(self.val_set_cases)
        if contamination:
            raise AssertionError(f'Contamination in test & val-set split -> {contamination}')

    def writer(
        self,
        data_type: str,
        set_type: str,
        case_name: str,
        file_type: str,
        array_data: np.array,
        h5_object: h5py.File,
    ) -> None:
        """Convert data to numpy array and write it hdf5 file"""
        new_file_path = f'{set_type}/{data_type}/{case_name}'
        group = h5_object.require_group(new_file_path)
        group.create_dataset(file_type, data=array_data)
        train_file_path = f'{new_file_path}/{file_type}'
        self.path_memory[f'{set_type}_dataset_paths'][data_type][case_name][file_type] = train_file_path
        # for 2D slice wise data
        # for slice_idx in range(0, np.shape(array_data)[1]):
        #     new_file_path = f'{set_type}/{data_type}/{case_name}/{slice_idx}'
        #     group = h5_object.require_group(new_file_path)
        #     group.create_dataset(file_type, data=array_data[:, slice_idx, :, :])
        #     train_file_path = f'{new_file_path}/{file_type}'
        #     self.path_memory[f'{set_type}_dataset_paths'][data_type][case_name][file_type][slice_idx] = train_file_path

    @staticmethod
    def load_data_type(file_path: str) -> np.array:
        """ "Read data as array"""
        return tio.ScalarImage(file_path).data

    def create_set(self, set_type: str, set_cases: str, h5_object: h5py.File) -> None:
        """Create data set and append location to path memory"""
        self.path_memory[f'{set_type}_dataset_paths'] = NestedDefaultDict()
        for data_type in ['data', 'label']:
            for case_name in set_cases:
                file_types = self.data_store_paths[data_type][case_name]
                for file_type in file_types:
                    file_path = self.data_store_paths[data_type][case_name][file_type]
                    array_data = self.load_data_type(file_path)
                    self.writer(data_type, set_type, case_name, file_type, array_data, h5_object)

    def execute_dataset_split(self) -> None:
        """Write files to train/validation/test folders in hdf5"""
        logger.info(f'Write hdf5 file -> {self.trainset_store_path}')
        with h5py.File(self.trainset_store_path, 'w') as h5_object:
            self.create_set('train', self.train_set_cases, h5_object)
            self.create_set('val', self.val_set_cases, h5_object)
            self.create_set('test', self.test_set_cases, h5_object)

    def branch_helper(self, name: str, obj: h5py.Group or h5py.Dataset) -> None:
        """Makes branches kinda pretty"""
        shift = name.count('/') * 3 * ' '  # convert / to and shift offset
        item_name = name.split('/')[-1]
        branch = f'{shift}{item_name}'
        if isinstance(obj, h5py.Dataset):
            branch = f'{branch} {obj.shape}'
        logger.info(branch)
        self.tree_store += f'{branch}\n'

    def show_tree_structure(self) -> None:
        """Visualize tree structure of hdf5"""
        h5_object = h5py.File(self.trainset_store_path, 'r')
        h5_object.visititems(self.branch_helper)  # iterates over each branch

    def store_tree_structure(self) -> None:
        """Export tree structure of hdf5"""
        tree_structure = os.path.join(self.params['project']['trainset_store_path'], 'data_structure.txt')
        with open(tree_structure, '+w') as f:
            f.write(self.tree_store)

