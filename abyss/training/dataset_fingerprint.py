import json
import os
from typing import List

import numpy as np
import SimpleITK as sitk
from loguru import logger
from monai.transforms import LoadImage


def compute_median_spacing(
    dataset_path: str,
    channel_order: dict,  # noqa: ARG001
    train_dataset_file: str,
) -> List[float]:
    """Compute median spacing from all training images.

    This function analyzes all training images to compute the median voxel spacing
    per axis, which is then used as the target spacing for resampling (nnUNet v2 approach).

    Args:
        dataset_path: Path to the training dataset directory
        channel_order: Dictionary mapping channel names to file suffixes
        train_dataset_file: Path to the JSON file containing training dataset list

    Returns:
        List of median spacings per axis [x, y, z] or [z, y, x] depending on convention

    Example:
        >>> spacing = compute_median_spacing(
        ...     '/path/to/train',
        ...     {'t1c': '_t1c.nii.gz'},
        ...     '/path/to/train_dataset.json'
        ... )
        >>> print(spacing)
        [1.0, 1.0, 1.0]
    """
    # Load training dataset
    with open(train_dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Get all training cases
    training_cases = dataset.get('training', [])

    if not training_cases:
        raise ValueError(f'No training cases found in {train_dataset_file}')

    spacings = []
    loader = LoadImage(image_only=False)  # Need metadata

    logger.info(f'Analyzing {len(training_cases)} training cases for spacing computation')

    for idx, case in enumerate(training_cases):
        subject_name = case['name']
        subject_path = os.path.join(dataset_path, subject_name)

        # Get first channel image for spacing
        channel_files = case['image']
        if not channel_files:
            continue

        # Load first available image
        image_path = os.path.join(subject_path, channel_files[0])

        if not os.path.exists(image_path):
            logger.warning(f'Image not found: {image_path}, skipping')
            continue

        try:
            # Load image with metadata
            img_data = loader(image_path)

            # Get spacing from metadata
            if hasattr(img_data, 'meta') and 'spacing' in img_data.meta:
                spacing = img_data.meta['spacing']
            else:
                # Fallback: use SimpleITK to read spacing directly
                sitk_img = sitk.ReadImage(image_path)
                spacing = sitk_img.GetSpacing()

            # Convert to list and handle orientation
            # MONAI typically uses [x, y, z] but SimpleITK uses [x, y, z]
            # We'll convert to list format consistent with config
            spacing_list = list(spacing) if isinstance(spacing, tuple) else spacing

            # Handle different axis orders (RAS vs others)
            # Assuming we want [z, y, x] or [x, y, z] - adjust based on your convention
            # For nnUNet compatibility, typically [z, y, x] or [x, y, z]
            # Reverse if needed to match config convention [z, y, x]
            if len(spacing_list) >= 3:
                # Most common: reverse to match [z, y, x] convention
                spacing_list = spacing_list[::-1] if len(spacing_list) == 3 else spacing_list

            spacings.append(spacing_list)

            if (idx + 1) % 10 == 0:
                logger.debug(f'Processed {idx + 1}/{len(training_cases)} cases')

        except Exception as e:
            logger.warning(f'Failed to load {image_path}: {e}, skipping')
            continue

    if not spacings:
        raise ValueError('No valid images found to compute spacing')

    # Compute median spacing per axis
    spacings_array = np.array(spacings)
    median_spacing = np.median(spacings_array, axis=0).tolist()

    logger.info(f'Computed median spacing: {median_spacing}')
    logger.debug('Spacing range per axis')
    for axis_idx, axis_name in enumerate(['Z', 'Y', 'X'][: len(median_spacing)]):
        min_sp = float(np.min(spacings_array[:, axis_idx]))
        max_sp = float(np.max(spacings_array[:, axis_idx]))
        logger.debug(f'  {axis_name}: {min_sp:.4f} - {max_sp:.4f} (median: {median_spacing[axis_idx]:.4f})')

    return median_spacing


def fingerprint_dataset(config: dict) -> dict:
    """Compute dataset fingerprint and update config with automatic parameters.

    This function analyzes the dataset to compute:
    - Median spacing (target spacing for resampling)
    - Other dataset properties (can be extended)

    Args:
        config: Configuration dictionary

    Returns:
        Updated config dictionary with fingerprinting results
    """
    config_path = config['project']['config_path']
    train_dataset_file = os.path.join(config_path, 'train_dataset.json')

    # Check if dataset file exists
    if not os.path.exists(train_dataset_file):
        logger.warning(f'Dataset file not found: {train_dataset_file}')
        logger.info('Using manual spacing from config')
        return config

    # Compute median spacing
    try:
        median_spacing = compute_median_spacing(
            dataset_path=config['project']['train_dataset_path'],
            channel_order=config['dataset']['channel_order'],
            train_dataset_file=train_dataset_file,
        )

        # Update config with computed spacing
        # Use 'target_spacing' for the resampled spacing, keep 'spacing' for backward compat
        config['dataset']['target_spacing'] = median_spacing

        logger.success('Dataset fingerprinting complete')
        logger.info(f'Target spacing: {median_spacing}')
        logger.info('(This will be used for resampling all images)')

    except Exception as e:
        logger.warning(f'Failed to compute median spacing: {e}')
        logger.info('Using manual spacing from config')
        # Fallback to manual spacing
        config['dataset']['target_spacing'] = config['dataset']['spacing']

    return config


if __name__ == '__main__':
    from config import ConfigFile

    config_dict = ConfigFile().get_config()
    config_dict = fingerprint_dataset(config_dict)
    logger.info(f'Final target spacing: {config_dict["dataset"].get("target_spacing", "NOT SET")}')
