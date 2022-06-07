import os

import h5py
import numpy as np
import SimpleITK as sitk
from loguru import logger


class CreateTrainset:
    """Create train, val, and test set for training"""

    def __init__(self, _config_manager):
        self.config_manager = _config_manager
        self.params = _config_manager.params
        self.preprocessed_store_paths = _config_manager.get_path_memory('preprocessed_dataset_paths')
        self.trainset_store_path = os.path.join(self.params['project']['trainset_store_path'], 'data.h5')
        self.train_set_cases = None
        self.val_set_cases = None
        self.test_set_cases = None
        np.random.seed(self.config_manager.params['meta']['seed'])

    def __call__(self):
        logger.info(f'Run: {self.__class__.__name__}')
        self.train_test_split()
        self.cross_fold_split()
        self.train_val_split()
        self.execute_dataset_split()
        self.config_manager.store_path_memory_file()
        self.show_tree_structure()

    def train_test_split(self):
        """Creates a list with case names for train and test set each"""
        count_cases = len(self.preprocessed_store_paths['image'])
        if count_cases < 10:
            raise AssertionError('Your dataset needs to have at least 10 subjects')
        test_set_size = int(self.params['dataset']['test_frac'] * count_cases)
        self.test_set_cases = list(
            np.random.choice(list(self.preprocessed_store_paths['image']), size=test_set_size, replace=False)
        )
        self.train_set_cases = [x for x in self.preprocessed_store_paths['image'] if x not in self.test_set_cases]
        if set(self.test_set_cases) & set(self.train_set_cases):
            raise AssertionError('Contamination in train & test-set split')
        logger.info(f'Test set, counts: {len(self.test_set_cases)}, cases: {self.test_set_cases}')

    def cross_fold_split(self):
        """Create cross fold split"""
        if '/' not in self.params['dataset']['cross_fold']:
            raise ValueError('config_file -> dataset -> cross_fold: fold_number/max_number_of_folds')
        max_folds = int(self.params['dataset']['cross_fold'].split('/')[1])
        if max_folds <= 0:
            raise ValueError('config_file -> dataset -> cross_fold: max_number_of_folds >= 1')
        fold_number = int(self.params['dataset']['cross_fold'].split('/')[0])
        if fold_number <= 0:
            raise ValueError('config_file -> dataset -> cross_fold: fold_number >= 1')
        fold_size = int(np.divide(len(self.train_set_cases), max_folds))
        count_remaining = int(len(self.train_set_cases) % max_folds)
        remainder = 0
        tmp_fold_cases = None
        for _ in range(1, fold_number + 1, 1):
            if count_remaining > 0:
                remainder = 1
                count_remaining -= 1
            tmp_fold_cases = list(np.random.choice(self.train_set_cases, size=fold_size + remainder, replace=False))
            remainder = 0
        self.train_set_cases = tmp_fold_cases

    def train_val_split(self):
        """Split train data into train and val data"""
        count_cases = len(self.train_set_cases)
        val_set_size = int(self.params['dataset']['val_frac'] * count_cases)
        self.val_set_cases = list(np.random.choice(self.train_set_cases, size=val_set_size, replace=False))
        self.train_set_cases = [x for x in self.train_set_cases if x not in self.val_set_cases]
        if set(self.train_set_cases) & set(self.val_set_cases):
            raise AssertionError('Contamination in train & val-set split')
        logger.info(f'Train set, counts: {len(self.train_set_cases)}, cases: {self.train_set_cases}')
        logger.info(f'Val set, counts: {len(self.val_set_cases)}, cases: {self.val_set_cases}')

    @staticmethod
    def writer(hf_object, group, set_type, case_name, file_tag, file_path):
        """Convert data to numpy array and write it hdf5 file"""
        img = sitk.ReadImage(file_path)
        img_arr = sitk.GetArrayFromImage(img)
        group = hf_object.require_group(f'{group}/{set_type}/{case_name}')
        group.create_dataset(file_tag, data=img_arr)

    def create_set(self, hf_object, set_cases, set_tag, data_type):
        """Create data set"""
        for case_name in set_cases:
            file_tags = self.preprocessed_store_paths[data_type][case_name]
            for file_tag in file_tags:
                file_path = self.preprocessed_store_paths[data_type][case_name][file_tag]
                self.writer(hf_object, set_tag, data_type, case_name, file_tag, file_path)
                self.config_manager.path_memory[
                    f'{set_tag}_dataset_paths'
                ] = f'{set_tag}/{data_type}/{case_name}/{file_tag}'

    def execute_dataset_split(self):
        """Copies files to folders: imageTr, labelTr, imageTs, labelTs"""
        with h5py.File(self.trainset_store_path, 'w') as hf_object:
            self.create_set(hf_object, self.train_set_cases, 'train', 'image')
            self.create_set(hf_object, self.train_set_cases, 'train', 'label')
            self.create_set(hf_object, self.val_set_cases, 'val', 'image')
            self.create_set(hf_object, self.val_set_cases, 'val', 'label')
            self.create_set(hf_object, self.test_set_cases, 'test', 'image')
            self.create_set(hf_object, self.test_set_cases, 'test', 'label')

    @staticmethod
    def branch_formatter(name, obj):
        """Format structure of hdf5 items"""
        shift = name.count('/') * 3 * ' '
        item_name = name.split('/')[-1]
        branch = f'{shift}{item_name}'
        if isinstance(obj, h5py.Dataset):
            branch = f'{branch} {obj.shape}'
        logger.info(branch)

    def show_tree_structure(self):
        """Visualize tree structure of hdf5"""
        hf_object = h5py.File(self.trainset_store_path, 'r')
        hf_object.visititems(self.branch_formatter)
