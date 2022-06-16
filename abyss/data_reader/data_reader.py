import json
import os

from loguru import logger

from abyss.config import ConfigManager
from abyss.data_reader.file_finder import FileFinder
from abyss.data_reader.restructure import Restructure
from abyss.utils import NestedDefaultDict, assure_instance_type


class DataReader(ConfigManager):
    """Read and clean original data"""

    def __init__(self, **kwargs):
        super().__init__()
        self._shared_state.update(kwargs)
        self.label_search_tags = assure_instance_type(self.params['dataset']['label_search_tags'], dict)
        self.data_search_tags = assure_instance_type(self.params['dataset']['data_search_tags'], dict)
        self.data_path_store = NestedDefaultDict()

    def __call__(self):
        """Run"""
        logger.info(f'Run: {self.__class__.__name__}')
        file_finder = FileFinder()
        self.data_path_store = file_finder()
        self.show_dict_findings()
        data_restruct = Restructure(self.data_path_store)
        data_restruct()
        self.store_path_memory_file()

    def show_dict_findings(self):
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
            'Total cases': len(self.data_path_store['data'].keys()),
            'Label': count_labels,
            'Data': count_data,
        }
        logger.info(f'Dataset scan overview: {json.dumps(stats_dict, indent=4)}')
