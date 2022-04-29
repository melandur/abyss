import os
import shutil

from loguru import logger

from abyss.utils import assure_instance_type


class DataRestruct:
    """Restructure original data"""

    def __init__(self, config_manager, data_path_store):
        self.config_manager = config_manager
        self.path_memory = config_manager.path_memory
        self.image_search_tags = assure_instance_type(config_manager.params['dataset']['image_search_tags'], dict)
        self.data_path_store = data_path_store

    def __call__(self):
        """Run data restruction"""
        logger.info(f'Run: {self.__class__.__name__}')
        self.create_structured_dataset()

    def create_structured_dataset(self):
        """Copy files from original dataset to structured dataset and create file path dict"""
        logger.info('Copying original dataset into structured dataset')

        for case_name in sorted(self.data_path_store['image']):
            for tag_name in self.image_search_tags:  # copy images
                self.path_memory['structured_dataset_paths']['image'][case_name][tag_name] = self.copy_helper(
                    src=self.data_path_store['image'][case_name][tag_name],
                    folder_name='image',
                    case_name=case_name,
                    tag_name=tag_name,
                )

            # copy labels
            self.path_memory['structured_dataset_paths']['label'][case_name] = self.copy_helper(
                src=self.data_path_store['label'][case_name],
                folder_name='label',
                case_name=case_name,
                tag_name='seg',
            )

        self.config_manager.store_path_memory_file()

    def copy_helper(self, src, folder_name, case_name, tag_name):
        """Copy and renames files by their case and tag name, keeps file extension, returns the new file path"""
        if isinstance(src, str) and os.path.isfile(src):
            file_name = os.path.basename(src)
            file_extension = file_name.split(os.extsep, 1)[1]  # split on first . and uses the rest as extension
            new_file_name = f'{case_name}_{tag_name}.{file_extension}'
            dst_file_path = os.path.join(
                self.config_manager.params['project']['structured_dataset_store_path'],
                folder_name,
                new_file_name,
            )
            os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
            shutil.copy2(src=src, dst=dst_file_path)
            return dst_file_path
        raise AssertionError(f'Files seems not to exist, case: {case_name}, tag: {tag_name}')
