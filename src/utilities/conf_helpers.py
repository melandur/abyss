import os
from copy import deepcopy
from collections import Counter
from loguru import logger as log


def check_and_create_folder_structure(params):
    """Check and create folders if they are missing"""
    folders = [
        params['project']['dataset_store_path'],
        params['project']['result_store_path'],
        params['project']['augmentation_store_path'],
        params['dataset']['folder_path']
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def check_image_search_tag_redundancy(params):
    """Check if there are any redundant search tag per image name"""
    for key, value in params['dataset']['image_search_tags'].items():
        if len(value) != len(set(value)):
            redundant_tag = list((Counter(value) - Counter(list(set(value)))).elements())
            log.error(f'The image search tag {redundant_tag} appears multiple times for the image name {key}')
            exit(1)


def check_image_search_tag_uniqueness(params):
    """Check if the image search tags are unique enough to avoid wrong data loading"""
    tags = [*params['dataset']['image_search_tags'].values()]
    tags = [x for sublist in tags for x in sublist]  # flatten nested list
    for i, tag in enumerate(tags):
        tmp_tags = deepcopy(tags)
        tmp_tags.pop(i)
        if [x for x in tmp_tags if x in tag]:
            vague_tag = [x for x in tmp_tags if x in tag]
            log.error(f'The image search tag {vague_tag} is not expressive/unique enough. '
                      f'Try to add additional information to the search tag like "_", "."')
            exit(1)
