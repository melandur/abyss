import os
import shutil

import numpy as np


class CreateTrainset:
    """Create train, val, and test set for training"""

    def __init__(self, _config_manager):
        self.config_manager = _config_manager
        self.params = _config_manager.params
        self.preprocessed_store_paths = _config_manager.get_path_memory('preprocessed_dataset_paths')

        self.train_set_cases = None
        self.val_set_cases = None
        self.test_set_cases = None

        np.random.seed(self.config_manager.params['meta']['seed'])

    def __call__(self):
        logger.info(f'Run: {self.__class__.__name__}')
        self.train_test_split_by_case_names()
        self.train_val_split_by_case_names()
        self.execute_dataset_split()
        self.config_manager.store_path_memory_file()

    def train_test_split_by_case_names(self):
        """Creates a list with case names for train and test set each"""
        count_cases = len(self.preprocessed_store_paths['image'])
        test_set_size = int(self.params['dataset']['test_frac'] * count_cases)
        self.test_set_cases = list(
            np.random.choice(list(self.preprocessed_store_paths['image']), size=test_set_size, replace=False)
        )
        self.train_set_cases = [x for x in self.preprocessed_store_paths['image'] if x not in self.test_set_cases]
        if set(self.test_set_cases) & set(self.train_set_cases):
            raise AssertionError('Contamination in train & test-set split')
        logger.info(f'Test set, counts: {len(self.test_set_cases)}, cases: {self.test_set_cases}')

    def train_val_split_by_case_names(self):
        """Split train data into train and val data"""
        count_cases = len(self.train_set_cases)
        val_set_size = int(self.params['dataset']['val_frac'] * count_cases)
        self.val_set_cases = list(np.random.choice(self.train_set_cases, size=val_set_size, replace=False))
        self.train_set_cases = [x for x in self.train_set_cases if x not in self.val_set_cases]

        if set(self.train_set_cases) & set(self.val_set_cases):
            raise AssertionError('Contamination in train & val-set split')

        logger.info(f'Train set, counts: {len(self.train_set_cases)}, cases: {self.train_set_cases}')
        logger.info(f'Val set, counts: {len(self.val_set_cases)}, cases: {self.val_set_cases}')

    def execute_dataset_split(self):
        """Copies files to folders: imageTr, labelTr, imageTs, labelTs"""

        def copy_helper(src, folder_name):
            file_name = os.path.basename(src)
            try:
                dst_file_path = os.path.join(self.params['project']['trainset_store_path'], folder_name, file_name)
                shutil.copy2(src, dst_file_path)
                return dst_file_path
            except Exception as error:
                raise error

        # copy train dataset
        for case_name in self.train_set_cases:
            image_tr = copy_helper(self.preprocessed_store_paths['image'][case_name], 'imagesTr')
            self.config_manager.path_memory['train_dataset_paths']['image'][case_name] = image_tr
            labels_tr = copy_helper(self.preprocessed_store_paths['label'][case_name], 'labelsTr')
            self.config_manager.path_memory['train_dataset_paths']['label'][case_name] = labels_tr

        # copy val dataset
        for case_name in self.val_set_cases:
            images_val = copy_helper(self.preprocessed_store_paths['image'][case_name], 'imagesVal')
            self.config_manager.path_memory['val_dataset_paths']['image'][case_name] = images_val
            labels_val = copy_helper(self.preprocessed_store_paths['label'][case_name], 'labelsVal')
            self.config_manager.path_memory['val_dataset_paths']['label'][case_name] = labels_val

        # copy test dataset
        for case_name in self.test_set_cases:
            images_ts = copy_helper(self.preprocessed_store_paths['image'][case_name], 'imagesTs')
            self.config_manager.path_memory['test_dataset_paths']['image'][case_name] = images_ts
            labels_ts = copy_helper(self.preprocessed_store_paths['label'][case_name], 'labelsTs')
            self.config_manager.path_memory['test_dataset_paths']['label'][case_name] = labels_ts


if __name__ == '__main__':
    import sys

    from loguru import logger

    from abyss.config import ConfigManager

    config_manager = ConfigManager(load_config_file_path=None)
    logger.remove()  # fresh start
    logger.add(sys.stderr, level=config_manager.params['logger']['level'])
    # for x, a in t.train_data_path_store['image'].items():
    #     print(x, a)
