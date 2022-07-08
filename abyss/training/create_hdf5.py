import copy
import os

import h5py
import numpy as np
import SimpleITK as sitk
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
        logger.info(f'Run: {self.__class__.__name__}')
        self.data_store_paths = self.get_data_store_paths()
        self.check_fold_settings()
        self.train_test_split()
        self.train_val_split()
        self.contamination_check()
        self.execute_dataset_split()
        self.show_tree_structure()
        self.store_path_memory_file()

    def get_data_store_paths(self) -> NestedDefaultDict:
        """Returns the current data store path, with prio 1: preprocessed, prio 2: structured dataset"""
        if len(self.path_memory['preprocessed_dataset_paths']['data']) != 0:
            logger.info('HDF5 file will be created from preprocessed dataset')
            return self.path_memory['preprocessed_dataset_paths']
        if len(self.path_memory['structured_dataset_paths']['data']) != 0:
            logger.info('HDF5 file will be created from structured dataset')
            return self.path_memory['structured_dataset_paths']
        raise ValueError(
            'Path memory file is empty for structured and preprocessed data, check config_file -> pipeline_steps'
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

    @staticmethod
    def writer(h5_object: h5py.File, group: str, set_type: str, case_name: str, file_tag: str, file_path) -> None:
        """Convert data to numpy array and write it hdf5 file"""
        img = sitk.ReadImage(file_path)
        img_arr = sitk.GetArrayFromImage(img)
        group = h5_object.require_group(f'{group}/{set_type}/{case_name}')
        group.create_dataset(file_tag, data=img_arr)

    def create_set(self, h5_object: h5py.File, set_cases: str, set_tag: str, data_type: str) -> None:
        """Create data set and append location to path memory"""
        for case_name in set_cases:
            file_tags = self.data_store_paths[data_type][case_name]
            for file_tag in file_tags:
                file_path = self.data_store_paths[data_type][case_name][file_tag]
                self.writer(h5_object, set_tag, data_type, case_name, file_tag, file_path)
                train_file_path = f'{set_tag}/{data_type}/{case_name}/{file_tag}'
                self.path_memory[f'{set_tag}_dataset_paths'][data_type][case_name][file_tag] = train_file_path

    def execute_dataset_split(self) -> None:
        """Write files to train/validation/test folders in hdf5"""
        logger.info(f'Write hdf5 file -> {self.trainset_store_path}')
        self.path_memory['train_dataset_paths'] = NestedDefaultDict()
        self.path_memory['val_dataset_paths'] = NestedDefaultDict()
        self.path_memory['test_dataset_paths'] = NestedDefaultDict()
        with h5py.File(self.trainset_store_path, 'w') as h5_object:
            self.create_set(h5_object, self.train_set_cases, 'train', 'data')
            self.create_set(h5_object, self.train_set_cases, 'train', 'label')
            self.create_set(h5_object, self.val_set_cases, 'val', 'data')
            self.create_set(h5_object, self.val_set_cases, 'val', 'label')
            self.create_set(h5_object, self.test_set_cases, 'test', 'data')
            self.create_set(h5_object, self.test_set_cases, 'test', 'label')

    def branch_helper(self, name: str, obj: h5py.Group or h5py.Dataset) -> None:
        """Makes branches kinda pretty"""
        shift = name.count('/') * 3 * ' '  # convert / to and shift offset
        item_name = name.split('/')[-1]
        branch = f'{shift}{item_name}'
        if isinstance(obj, h5py.Dataset):
            branch = f'{branch} {obj.shape}'
        self.tree_store += f'\n{branch}'

    def show_tree_structure(self) -> None:
        """Visualize tree structure of hdf5"""
        h5_object = h5py.File(self.trainset_store_path, 'r')
        h5_object.visititems(self.branch_helper)  # iterates over each branch
        logger.info(self.tree_store)
