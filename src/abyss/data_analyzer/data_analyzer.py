import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torchio as tio
from loguru import logger
from numpyencoder import NumpyEncoder

from abyss.utils import NestedDefaultDict


class DataAnalyzer:
    """Some basic dataset analyser, whole dataset as case wise"""

    def __init__(self, params: dict, path_memory: dict) -> None:
        self.params = params
        self.path_memory = path_memory
        self.stats_cases = NestedDefaultDict()
        self.stats_dataset = NestedDefaultDict()
        self.hist_bins = 10

    def __call__(self, state: str) -> None:
        if self.params['pipeline_steps']['data_reader']:
            logger.info(f'Run: {self.__class__.__name__} -> {state}')
            for case_name in self.path_memory[f'{state}_paths']['data']:
                for data_type in self.path_memory[f'{state}_paths']['data'][case_name]:
                    file_path = self.path_memory[f'{state}_paths']['data'][case_name][data_type]
                    self.analyse_case(case_name, data_type, file_path)
                    self.format_output(self.stats_cases, case_name, data_type)

            self.analyse_dataset()
            export_folder = os.path.join(self.params['project'][f'{state}_store_path'], 'stats')
            os.makedirs(export_folder, exist_ok=True)
            self.export_stats(export_folder)
            self.export_dataset_plots(export_folder)

    def analyse_case(self, case_name: str, data_type: str, file_path: str) -> None:
        """Analyse data case wise"""
        data = self.read_file(file_path)
        self.stats_cases[case_name][data_type]['type'] = data.type
        self.stats_cases[case_name][data_type]['origin'] = data.origin
        self.stats_cases[case_name][data_type]['spacing'] = data.spacing
        self.stats_cases[case_name][data_type]['direction'] = data.direction
        self.stats_cases[case_name][data_type]['orientation'] = data.orientation
        self.stats_cases[case_name][data_type]['spatial_shape'] = data.spatial_shape
        data_arr = data.numpy()
        self.stats_cases[case_name][data_type]['min'] = np.min(data_arr)
        self.stats_cases[case_name][data_type]['max'] = np.max(data_arr)
        self.stats_cases[case_name][data_type]['std'] = np.std(data_arr)
        self.stats_cases[case_name][data_type]['mean'] = np.mean(data_arr)
        self.stats_cases[case_name][data_type]['median'] = np.median(data_arr)
        hist, bin_edges = np.histogram(data_arr, bins=self.hist_bins)
        self.stats_cases[case_name][data_type]['hist'] = hist
        self.stats_cases[case_name][data_type]['bin_edges'] = bin_edges

    @staticmethod
    def read_file(file_path: str) -> np.array:
        """Read file path as array"""
        return tio.ScalarImage(file_path)

    @staticmethod
    def format_output(stats_cases: NestedDefaultDict, case_name: str, data_type: str) -> None:
        """Reduce the output, no need to flood the terminal"""
        show_keys = ['spatial_shape', 'origin', 'spacing', 'orientation', 'min', 'max', 'std']
        tmp_store = {}
        for key, value in stats_cases[case_name][data_type].items():
            if key in show_keys:
                tmp_store[key] = value
        logger.trace(f'-> {case_name} -> {data_type} -> {json.dumps(tmp_store, indent=4, cls=NumpyEncoder)}')

    def analyse_dataset(self) -> None:
        """Analyse the whole dataset"""
        tmp_min, tmp_max, tmp_mean, tmp_median, tmp_std, tmp_hist, tmp_edges = [], [], [], [], [], [], []
        counter = 0
        for case_name in self.stats_cases:
            for data_type in self.stats_cases[case_name]:
                counter += 1
                tmp_min.append(self.stats_cases[case_name][data_type]['min'])
                tmp_max.append(self.stats_cases[case_name][data_type]['max'])
                tmp_mean.append(self.stats_cases[case_name][data_type]['mean'])
                tmp_median.append(self.stats_cases[case_name][data_type]['median'])
                tmp_std.append(self.stats_cases[case_name][data_type]['std'])
                tmp_hist.append(self.stats_cases[case_name][data_type]['hist'])
                tmp_edges.append(self.stats_cases[case_name][data_type]['bin_edges'])

        self.stats_dataset['cases'] = len(self.stats_cases)
        self.stats_dataset['min'] = np.min(tmp_min)
        self.stats_dataset['max'] = np.max(tmp_max)
        self.stats_dataset['mean'] = np.mean(tmp_mean)
        self.stats_dataset['median'] = np.median(tmp_median)
        self.stats_dataset['std'] = np.std(tmp_std)
        self.stats_dataset['hist'], _ = np.histogram(np.sum(tmp_hist, axis=0) / counter, bins=self.hist_bins)
        _, self.stats_dataset['bin_edges'] = np.histogram(np.sum(tmp_edges, axis=0) / counter, bins=self.hist_bins)

    def export_dataset_plots(self, export_folder: str) -> None:
        """Plot dataset histogram"""
        bar_width = (self.stats_dataset['mean'] - self.stats_dataset['min']) / self.hist_bins
        plt.bar(self.stats_dataset['bin_edges'][:-1], self.stats_dataset['hist'], width=bar_width)
        plt.title('Histogram dataset, 10 bins')
        plt.xlabel('Intensities')
        plt.ylabel('Counts')
        plt.savefig(os.path.join(export_folder, 'histogram.png'))
        plt.close()

    def export_stats(self, export_folder: str) -> None:
        """Exports stats as json"""
        file_path_dataset = os.path.join(export_folder, 'dataset.json')
        with open(file_path_dataset, 'w+', encoding='utf-8') as file_object:
            json.dump(self.stats_dataset, file_object, indent=4, cls=NumpyEncoder)
        file_path_cases = os.path.join(export_folder, 'cases.json')
        with open(file_path_cases, 'w+', encoding='utf-8') as file_object:
            json.dump(self.stats_cases, file_object, indent=4, cls=NumpyEncoder)
