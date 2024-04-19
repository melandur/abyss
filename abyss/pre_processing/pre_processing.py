import os
from copy import deepcopy

import numpy as np
import SimpleITK as sitk
from loguru import logger
from scipy.ndimage import binary_fill_holes

from abyss.config import ConfigManager
from abyss.utils import NestedDefaultDict


class PreProcessing(ConfigManager):
    """Preprocess data/labels"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)
        self.image_boundaries = {}
        self.case = None
        self.file_path = None
        self.pre_precessed_file_path = None
        np.random.seed(self.params['meta']['seed'])

    def __call__(self) -> None:
        if self.params['pipeline_steps']['pre_processing']:
            logger.info(f'Run: {self.__class__.__name__}')

            self.path_memory['pre_processed_dataset_paths'] = NestedDefaultDict()
            self.get_image_boundaries()

            for case, data_type, group, tag, file_path in self.path_memory_iter(step='structured_dataset_paths'):
                self.case = case
                self.file_path = file_path
                tmp_img = None
                self.pre_precessed_file_path = self.get_pre_processed_file_path(case, data_type, group, tag, file_path)

                pre_process = self.params['pre_processing'][data_type][group]
                for process, params in pre_process.items():
                    params_ = deepcopy(params)
                    if hasattr(self, process):
                        if params_['active']:
                            logger.trace(f'pre_processing -> {process}')
                            params_.pop('active', None)
                            method = getattr(self, process)
                            tmp_img = method(tmp_img, params_)
                    else:
                        logger.warning(f'Pre-processing method not found: {process}')

        self.store_path_memory_file()

    def get_pre_processed_file_path(self, case, data_type, group, tag, file_path):
        """Get pre-processed file path"""
        pre_processed_dataset_store_path = self.params['project']['pre_processed_dataset_store_path']
        pre_precessed_folder_path = str(os.path.join(pre_processed_dataset_store_path, case, data_type, group))
        os.makedirs(pre_precessed_folder_path, exist_ok=True)
        file_name = os.path.basename(file_path)
        pre_processed_file_path = os.path.join(pre_precessed_folder_path, file_name)
        self.path_memory['pre_processed_dataset_paths'][case][data_type][group][tag] = pre_processed_file_path
        return pre_processed_file_path

    def simple_itk_reader(self, _, params: dict) -> sitk.Image:
        """Reads file with SimpleITK, if orientation is given, it orients the image accordingly"""
        img = sitk.ReadImage(self.file_path)
        img = sitk.Cast(img, eval(params['file_type']))
        if params['orientation']:
            filter_ = sitk.DICOMOrientImageFilter()
            filter_.SetDesiredCoordinateOrientation(params['orientation'])
            return filter_.Execute(img)
        return img

    def simple_itk_writer(self, img: sitk.Image, params: dict) -> None:
        """Writes image with SimpleITK"""
        img = sitk.Cast(img, eval(params['file_type']))
        sitk.WriteImage(img, str(self.pre_precessed_file_path))

    @staticmethod
    def z_score_norm(img: sitk.Image, _) -> sitk.Image:
        """Z-score normalization"""
        img_arr = sitk.GetArrayFromImage(img)
        img_norm_arr = (img_arr - np.mean(img_arr)) / np.std(img_arr)
        img_norm_sitk = sitk.GetImageFromArray(img_norm_arr)
        img_norm_sitk.CopyInformation(img)
        return img_norm_sitk

    @staticmethod
    def clip_percentiles(img: sitk.Image, params: dict) -> sitk.Image:
        """Clips image intensities between percentiles"""
        img_arr = sitk.GetArrayFromImage(img)
        lower, upper = np.percentile(img_arr, [params['lower'], params['upper']])
        img_clipped_arr = np.clip(img_arr, lower, upper)
        img_clipped_sitk = sitk.GetImageFromArray(img_clipped_arr)
        img_clipped_sitk.CopyInformation(img)
        return img_clipped_sitk

    def resize(self, img: sitk.Image, params: dict) -> sitk.Image:
        """Resize image"""
        original_size = img.GetSize()
        resampling_factor = [sz / float(ns) for sz, ns in zip(original_size, params['dim'])]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(resampling_factor)
        resampler.SetSize(params['dim'])
        resampler.SetInterpolator(eval(params['interpolator']))
        resampled_image = resampler.Execute(img)
        return resampled_image

    @staticmethod
    def relabel_mask(img: sitk.Image, params: dict) -> sitk.Image:
        """Relabel image"""
        img_arr = sitk.GetArrayFromImage(img)
        relabel_img_arr = np.zeros_like(img_arr)
        for old_label, new_label in params['label_dict'].items():
            relabel_img_arr[img_arr == old_label] = new_label
        relabel_img_sitk = sitk.GetImageFromArray(relabel_img_arr)
        relabel_img_sitk.CopyInformation(img)
        return relabel_img_sitk

    @staticmethod
    def __crop_zeros_helper(img: sitk.Image) -> tuple:
        """Zero crop image"""
        img_arr = sitk.GetArrayFromImage(img)
        binary_mask = (img_arr > 0).astype(int)
        bool_mask = binary_fill_holes(binary_mask)
        binary_mask = np.where(bool_mask != 0)
        min_indices = [np.min(indices) for indices in binary_mask]
        max_indices = [np.max(indices) for indices in binary_mask]
        return min_indices, max_indices

    def get_image_boundaries(self) -> None:
        """Get image boundaries for zero cropping"""
        logger.info('Get image boundaries for zero cropping')
        for case, data_type, group, _, file_path in self.path_memory_iter(step='structured_dataset_paths'):
            self.file_path = file_path
            pre_process = self.params['pre_processing'][data_type][group]

            for process, params in pre_process.items():
                if 'reader' not in process:
                    continue
                params_ = deepcopy(params)

                if hasattr(self, process) and params_['active']:
                    method = getattr(self, process)
                    params_.pop('active', None)
                    img = method(None, params_)
                    new_min_indices, new_max_indices = self.__crop_zeros_helper(img)
                    if self.image_boundaries.get(case, None):
                        old_min_indices, old_max_indices = self.image_boundaries[case]
                        min_values = tuple(min(new, old) for new, old in zip(new_min_indices, old_min_indices))
                        max_values = tuple(max(new, old) for new, old in zip(new_max_indices, old_max_indices))
                        self.image_boundaries[case] = (min_values, max_values)
                    else:
                        self.image_boundaries[case] = (new_min_indices, new_max_indices)

    def crop_zeros(self, img: sitk.Image, _) -> sitk.Image:
        """Zero crop image"""
        img_arr = sitk.GetArrayFromImage(img)
        min_indices, max_indices = self.image_boundaries[self.case]
        bbox = tuple(slice(min_idx, max_idx + 1) for min_idx, max_idx in zip(min_indices, max_indices))
        crop_img_arr = img_arr[bbox]
        crop_img_sitk = sitk.GetImageFromArray(crop_img_arr)
        crop_img_sitk.SetOrigin(img.GetOrigin())
        crop_img_sitk.SetSpacing(img.GetSpacing())
        crop_img_sitk.SetDirection(img.GetDirection())
        crop_img_sitk.SetOrigin(img.GetOrigin())
        return crop_img_sitk
