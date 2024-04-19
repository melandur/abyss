import os
import shutil

from loguru import logger

from abyss.config import ConfigManager
from abyss.utils import NestedDefaultDict


class DataReader(ConfigManager):
    """Read and clean original data"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)
        self.data_description = self.params['dataset']['description']
        self.data_path_store = NestedDefaultDict()

    def __call__(self) -> None:
        """Run"""
        if self.params['pipeline_steps']['data_reader']:
            logger.info(f'Run: {self.__class__.__name__}')
            self.path_memory['structured_dataset_paths'] = NestedDefaultDict()
            self._read_data()
            self._check_for_missing_files()
            self._create_structured_dataset()
            self.store_path_memory_file()

    def __data_description_iter(self) -> tuple:
        """Iterate over data description"""
        for data_type, groups in self.data_description.items():
            for group, tags in groups.items():
                for tag, tag_filter in tags.items():
                    yield data_type, group, tag, tag_filter

    def __data_path_store_iter(self) -> tuple:
        """Iterate over data path store"""
        for data_type, cases in self.data_path_store.items():
            for case, groups in cases.items():
                for group, tags in groups.items():
                    for tag, file_path in tags.items():
                        yield data_type, case, group, tag, file_path

    def _read_data(self) -> None:
        """Read data"""
        dataset_path = self.params['project']['dataset_folder_path']
        cases = os.listdir(dataset_path)
        cases.sort()

        for case in cases:
            files = os.listdir(os.path.join(self.params['project']['dataset_folder_path'], case))
            files.sort()
            for file in files:
                file_path = os.path.join(dataset_path, case, file)
                for data_type, group, tag, tag_filter in self.__data_description_iter():
                    if file.endswith(tag_filter):
                        self.data_path_store[data_type][case][group][tag] = file_path

    def _create_structured_dataset(self) -> None:
        """Copy files from original dataset to structured dataset and create file path dict"""
        logger.info('Copying original data to new structure -> 2_pre_processed_dataset')
        for data_type, case, group, tag, ori_file_path in self.__data_path_store_iter():
            structured_data_store_path = self.params['project']['structured_dataset_store_path']
            dst_folder_path = os.path.join(structured_data_store_path, data_type, case, group)
            dst_file_path = os.path.join(dst_folder_path, os.path.basename(ori_file_path))
            os.makedirs(dst_folder_path, exist_ok=True)
            shutil.copy2(src=ori_file_path, dst=dst_file_path)
            self.path_memory['structured_dataset_paths'][data_type][case][group][tag] = dst_file_path

    def _check_for_missing_files(self) -> None:
        """Check if there are any data/label files are missing"""
        for data_type, group, tag, _ in self.__data_description_iter():
            for _, case, _, _, _ in self.__data_path_store_iter():
                if not isinstance(self.data_path_store[data_type][case][group][tag], str):
                    raise FileNotFoundError(f'No {tag} file found for case {case}')
