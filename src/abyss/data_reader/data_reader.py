import json
import os

from loguru import logger

from abyss.config import ConfigManager
from abyss.data_analyzer import DataAnalyzer
from abyss.data_reader.file_finder import FileFinder
from abyss.data_reader.restructure import Restructure
from abyss.utils import NestedDefaultDict, assure_instance_type


class DataReader(ConfigManager):
    """Read and clean original data"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)
        self.label_search_tags = assure_instance_type(self.params['dataset']['label_search_tags'], dict)
        self.data_search_tags = assure_instance_type(self.params['dataset']['data_search_tags'], dict)
        self.data_path_store = NestedDefaultDict()

    def __call__(self) -> None:
        """Run"""
        if self.params['pipeline_steps']['data_reader']:
            logger.info(f'Run: {self.__class__.__name__}')
            self.path_memory['structured_dataset_paths'] = NestedDefaultDict()
            file_finder = FileFinder()
            self.data_path_store = file_finder()
            self.show_dict_findings()
            data_restruct = Restructure(self.data_path_store)
            data_restruct()
            self.store_path_memory_file()
            DataAnalyzer(self.params, self.path_memory)('structured_dataset')

    def show_dict_findings(self) -> None:
        """Summaries the findings"""
        logger.trace(f'Dataset scan found: {json.dumps(self.data_path_store, indent=4)}')
        count_labels = {}
        for label_tag in self.label_search_tags.keys():
            count_labels[label_tag] = 0
        count_data = {}
        for data_tag in self.data_search_tags.keys():
            count_data[data_tag] = 0

        for case in self.data_path_store['data'].keys():
            for data_tag, data_path in self.data_path_store['data'][case].items():
                if os.path.isfile(data_path):
                    count_data[data_tag] += 1

        for case in self.data_path_store['label'].keys():
            for label_tag, label_path in self.data_path_store['label'][case].items():
                if os.path.isfile(label_path):
                    count_labels[label_tag] += 1

        stats_dict = {
            'Total files': sum(count_data.values()),
            'Label': count_labels,
            'Data': count_data,
        }
        logger.info(f'Dataset scan overview: {json.dumps(stats_dict, indent=4)}')

        n_data = sum(count_data.values())
        n_label = sum(count_labels.values())
        if n_data == 0:
            raise ValueError('Data not found, check -> config_file -> dataset -> data_search_tags')
        if n_label == 0:
            raise ValueError('Labels not found, check -> config_file -> dataset -> label_search_tags')
        if n_data % n_label != 0:
            raise ValueError(
                'Not every label has the same multiple of data, check for missing data or adapt search tags',
            )
