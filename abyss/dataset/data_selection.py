import json
import os

from loguru import logger
from typing_extensions import ClassVar

from abyss.dataset.data_analyzer import DataAnalyzer
# from abyss.dataset.data_restruct import DataRestruct
from abyss.dataset.layout_parser import LayoutParser
from abyss.utils import NestedDefaultDict, assure_instance_type


class DataSelection:
    """Read and clean original data"""

    def __init__(self, config_manager: ClassVar):
        self.config_manager = config_manager
        self.params = config_manager.params
        self.path_memory = config_manager.path_memory
        self.image_search_tags = assure_instance_type(self.params['dataset']['image_search_tags'], dict)

        self.data_path_store = NestedDefaultDict

        self.layout_parser = LayoutParser(self.params)
        self.layout_parser()

    def __call__(self):
        """Run"""
        logger.info(f'Run: {self.__class__.__name__}')
        data_analyzer = DataAnalyzer(self.config_manager)
        data_analyzer()
        self.data_path_store = data_analyzer.data_path_store
        self.layout_parser.decode_folder_layout_of_found_files(self.data_path_store)
        # self.data_path_store = self.layout_parser
        # self.show_dict_findings()
        # data_restruct = DataRestruct(self.config_manager, data_analyzer.data_path_store)
        # data_restruct()

    def show_dict_findings(self):
        """Summaries and shows the findings"""
        logger.trace(f'Dataset scan found: {json.dumps(self.data_path_store, indent=4)}')

        count_labels = 0
        count_images = {}
        for image_tag in self.image_search_tags.keys():
            count_images[image_tag] = 0

        for case in self.data_path_store['image'].keys():
            for image_tag, image_path in self.data_path_store['image'][case].items():
                if os.path.isfile(image_path):
                    count_images[image_tag] += 1

        for _, label_path in self.data_path_store['label'].items():
            if os.path.isfile(label_path):
                count_labels += 1

        stats_dict = {
            'Total cases': len(self.data_path_store['image'].keys()),
            'Labels': count_labels,
            'Images': count_images,
        }

        logger.info(f'Dataset scan overview: {json.dumps(stats_dict, indent=4)}')
