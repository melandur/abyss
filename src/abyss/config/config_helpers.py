import os
from collections import Counter
from copy import deepcopy


def check_and_create_folder_structure(params: dict) -> None:
    """Check and create folders if they are missing"""
    folders = [
        params['project']['config_store_path'],
        params['project']['structured_dataset_store_path'],
        params['project']['pre_processed_dataset_store_path'],
        params['project']['trainset_store_path'],
        params['project']['result_store_path'],
        params['project']['production_store_path'],
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def check_search_tag_redundancy(params: dict, data_type: str) -> None:
    """Check if there are any redundant search tag per data/label name"""
    for key, value in params['dataset'][f'{data_type}_search_tags'].items():
        if len(value) != len(set(value)):
            redundant_tag = list((Counter(value) - Counter(list(set(value)))).elements())
            raise ValueError(f'The {data_type} search tag {redundant_tag} found multiple times for the name {key}')


def check_search_tag_uniqueness(params: dict, data_type: str) -> None:
    """Check if the data/label search tags are unique enough to avoid wrong loading"""
    tags = params['dataset'][f'{data_type}_search_tags'].values()
    tags = [x for sublist in tags for x in sublist]  # flatten nested list
    for i, tag in enumerate(tags):
        tmp_tags = deepcopy(tags)
        tmp_tags.pop(i)
        if [x for x in tmp_tags if x in tag]:
            vague_tag = [x for x in tmp_tags if x in tag]
            raise ValueError(
                f'The {data_type} search tag {vague_tag} is not expressive/unique enough. '
                f'Try to add additional information to the search tag like " ", ".", "_"'
            )
