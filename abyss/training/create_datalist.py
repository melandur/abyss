import json
import os

import numpy as np
from sklearn.model_selection import KFold


def create_test_dataset_file(config):
    """Create a dataset file with test data."""
    dataset_path = config['project']['test_dataset_path']

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'dataset path not found -> {dataset_path}')

    subjects = os.listdir(dataset_path)

    channel_order = config['dataset']['channel_order']
    label_order = config['dataset']['label_order']

    datalist = {'test': []}
    for subject in subjects:
        files = os.listdir(os.path.join(dataset_path, subject))
        channel_list = []
        label_list = []

        for name, identifier in channel_order.items():
            for file in files:
                if file.endswith(identifier):
                    channel_list.append(file)

        for name, identifier in label_order.items():
            for file in files:
                if file.endswith(identifier):
                    label_list.append(file)
                    break

        if len(channel_list) != len(channel_order):
            raise FileNotFoundError(f'channel files not found for subject -> {subject}')

        if len(label_list) != len(label_order):
            raise FileNotFoundError(f'label files not found for subject -> {subject}')

        datalist['training'].append(
            {
                'name': subject,
                'image': channel_list,
                'label': label_list,
            }
        )

    config_path = config['project']['config_path']
    os.makedirs(config_path, exist_ok=True)
    dataset_file_path = os.path.join(config_path, 'test_dataset.json')
    with open(dataset_file_path, 'w') as f:
        json.dump(datalist, f, indent=4)
    print(f'dataset file has been created -> {dataset_file_path}')


def create_train_dataset_file(config):
    """Create a dataset file with training data."""
    dataset_path = config['project']['train_dataset_path']

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'dataset path not found -> {dataset_path}')

    subjects = os.listdir(dataset_path)

    channel_order = config['dataset']['channel_order']
    label_order = config['dataset']['label_order']

    datalist = {'training': []}
    for subject in subjects:
        files = os.listdir(os.path.join(dataset_path, subject))
        channel_list = []
        label_list = []

        for name, identifier in channel_order.items():
            for file in files:
                if file.endswith(identifier):
                    channel_list.append(file)

        for name, identifier in label_order.items():
            for file in files:
                if file.endswith(identifier):
                    label_list.append(file)
                    break

        if len(channel_list) != len(channel_order):
            raise FileNotFoundError(f'channel files not found for subject -> {subject}')

        if len(label_list) != len(label_order):
            raise FileNotFoundError(f'label files not found for subject -> {subject}')

        datalist['training'].append(
            {
                'name': subject,
                'image': channel_list,
                'label': label_list,
            }
        )

    config_path = config['project']['config_path']
    os.makedirs(config_path, exist_ok=True)
    dataset_file_path = os.path.join(config_path, 'train_dataset.json')
    with open(dataset_file_path, 'w') as f:
        json.dump(datalist, f, indent=4)
    print(f'dataset file has been created -> {dataset_file_path}')


def create_inference_dataset_file(config):
    """Create a dataset file with test data."""
    dataset_path = '/home/melandur/Downloads/ucsf_corr/ucsf_images'
    datalist_path = '/home/melandur/Downloads/ucsf_corr'

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'dataset path not found -> {dataset_path}')

    subjects = os.listdir(dataset_path)

    channel_order = config['dataset']['channel_order']

    datalist = {'inference': []}
    for subject in subjects:
        files = os.listdir(os.path.join(dataset_path, subject))
        channel_list = []

        for name, identifier in channel_order.items():
            for file in files:
                if file.endswith(identifier):
                    channel_list.append(file)

        if len(channel_list) != len(channel_order):
            raise FileNotFoundError(f'channel files not found for subject -> {subject}')

        datalist['inference'].append(
            {
                'name': subject,
                'image': channel_list,
            }
        )

    dataset_file_path = os.path.join(datalist_path, 'inference_dataset.json')
    with open(dataset_file_path, 'w') as f:
        json.dump(datalist, f, indent=4)
    print(f'dataset file has been created -> {dataset_file_path}')


def create_datalist(config):
    """Create a dataset file with folds."""
    dataset_file_path = os.path.join(config['project']['config_path'], 'train_dataset.json')

    with open(dataset_file_path, 'r') as f:
        dataset = json.load(f)

    dataset_with_folds = dataset.copy()

    names = [line['name'] for line in dataset['training']]
    # keys = [line['image'] for line in dataset['training']]
    dataset_train_dict = dict(zip(names, dataset['training']))
    all_names_sorted = np.sort(names)

    kfold = KFold(
        n_splits=config['dataset']['total_folds'],
        shuffle=True,
        random_state=config['dataset']['seed'],
    )
    for i, (train_idx, test_idx) in enumerate(kfold.split(all_names_sorted)):
        val_data = []
        train_data = []
        train_keys = np.array(all_names_sorted)[train_idx]
        test_keys = np.array(all_names_sorted)[test_idx]
        for key in test_keys:
            val_data.append(dataset_train_dict[key])
        for key in train_keys:
            train_data.append(dataset_train_dict[key])

        dataset_with_folds[f'val_fold_{i}'] = val_data
        dataset_with_folds[f'train_fold_{i}'] = train_data
    del dataset

    print(json.dumps(dataset_with_folds, indent=4))

    with open(dataset_file_path, 'w') as f:
        json.dump(dataset_with_folds, f, indent=4)

    print(f'dataset file with folds has been created -> {dataset_file_path}')


if __name__ == '__main__':
    from abyss.config import ConfigFile

    config = ConfigFile().get_config()
    create_train_dataset_file(config)
    create_datalist(config)

    # create_test_dataset_file(config)
    # create_inference_dataset_file(config)
