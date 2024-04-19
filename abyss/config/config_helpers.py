import os
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


def __helper_get_last_values(d):
    """Get the last values of a nested dict"""
    if isinstance(d, dict):
        if not d:
            return []
        last_values = []
        for value in d.values():
            last_values.extend(__helper_get_last_values(value))
        return last_values
    return [d]


def check_search_tag_uniqueness(params: dict) -> None:
    """Check if the data/label search tags are unique enough to avoid wrong loading"""
    data_description = params['dataset']['description']
    tag_filters = __helper_get_last_values(data_description)
    for i, tag in enumerate(tag_filters):  # check if the tags are unique enough
        tmp_tags = deepcopy(tag_filters)
        tmp_tags.pop(i)
        if [x for x in tmp_tags if x in tag]:
            vague_filter = [x for x in tmp_tags if x in tag][0]
            raise ValueError(
                f'The search tag "{vague_filter}" is not expressive/unique enough. '
                f'Try to add additional information to the search tag like " ", ".", "_"'
            )


def check_pipeline_steps(params: dict) -> None:
    """Check for training & production collision"""
    if any(params['pipeline_steps']['training'].values()) and any(params['pipeline_steps']['production'].values()):
        raise ValueError(
            'Current pipeline_steps are invalid: deactivate -> config_file -> pipeline_steps ->'
            ' "training" or "production" step'
        )
