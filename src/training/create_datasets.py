import os
import shutil
import numpy as np
from loguru import logger as log


class CreateDatasets:
    """Create train, val, and test set for training"""

    def __init__(self, cm):
        self.cm = cm
        self.params = cm.params
        self.preprocessed_store_paths = cm.get_path_memory('preprocessed_dataset_paths')

        self.train_set_cases = None
        self.val_set_cases = None
        self.test_set_cases = None

        np.random.seed(cm.params['dataset']['seed'])
        log.info(f'Init: {self.__class__.__name__}')

        self.train_test_split_by_case_names()
        self.train_val_split_by_case_names()
        self.execute_dataset_split()
        self.cm.store_path_memory_file()

    @log.catch
    def train_test_split_by_case_names(self):
        """Creates a list with case names for train and test set each"""
        count_cases = len(list(self.preprocessed_store_paths['image'].keys()))
        test_set_size = int(self.params['dataset']['test_frac'] * count_cases)
        self.test_set_cases = list(np.random.choice(list(self.preprocessed_store_paths['image']),
                                                    size=test_set_size,
                                                    replace=False))
        self.train_set_cases = [x for x in list(self.preprocessed_store_paths['image'].keys()) if
                                x not in self.test_set_cases]
        if set(self.test_set_cases) & set(self.train_set_cases):
            log.error('Contamination in train & test-set split'), exit(1)
        log.info(f'Test set, counts: {len(self.test_set_cases)}, cases: {self.test_set_cases}')

    @log.catch
    def train_val_split_by_case_names(self):
        """Split train data into train and val data"""
        count_cases = len(self.train_set_cases)
        val_set_size = int(self.params['dataset']['val_frac'] * count_cases)
        self.val_set_cases = list(np.random.choice(self.train_set_cases,
                                                   size=val_set_size,
                                                   replace=False))
        self.train_set_cases = [x for x in self.train_set_cases if x not in self.val_set_cases]

        if set(self.train_set_cases) & set(self.val_set_cases):
            log.error('Contamination in train & val-set split'), exit(1)

        log.info(f'Train set, counts: {len(self.train_set_cases)}, cases: {self.train_set_cases}')
        log.info(f'Val set, counts: {len(self.val_set_cases)}, cases: {self.val_set_cases}')

    @log.catch
    def execute_dataset_split(self):
        """Copies files to folders: imageTr, labelTr, imageTs, labelTs"""

        def copy_helper(src, folder_name):
            file_name = os.path.basename(src)
            try:
                dst_file_path = os.path.join(self.params['project']['trainset_store_path'], folder_name, file_name)
                shutil.copy2(src, dst_file_path)
                return dst_file_path
            except Exception as e:
                log.error(e), exit(1)

        # copy train dataset
        for case_name in self.train_set_cases:
            self.cm.path_memory['train_dataset_paths']['image'][case_name] = copy_helper(
                self.preprocessed_store_paths['image'][case_name], 'imagesTr')
            self.cm.path_memory['train_dataset_paths']['label'][case_name] = copy_helper(
                self.preprocessed_store_paths['label'][case_name], 'labelsTr')

        # copy val dataset
        for case_name in self.val_set_cases:
            self.cm.path_memory['val_dataset_paths']['image'][case_name] = copy_helper(
                self.preprocessed_store_paths['image'][case_name], 'imagesVal')
            self.cm.path_memory['val_dataset_paths']['label'][case_name] = copy_helper(
                self.preprocessed_store_paths['label'][case_name], 'labelsVal')

        # copy test dataset
        for case_name in self.test_set_cases:
            self.cm.path_memory['test_dataset_paths']['image'][case_name] = copy_helper(
                self.preprocessed_store_paths['image'][case_name], 'imagesTs')
            self.cm.path_memory['test_dataset_paths']['label'][case_name] = copy_helper(
                self.preprocessed_store_paths['label'][case_name], 'labelsTs')


if __name__ == '__main__':
    import sys
    from loguru import logger as log
    from src.config.config_manager import ConfigManager

    params = ConfigManager(load_conf_file_path=None).params
    log.remove()  # fresh start
    log.add(sys.stderr, level=params['logger']['level'])
    t = TrainDataSetPathScan(params)
    # for x, a in t.train_data_path_store['image'].items():
    #     print(x, a)
