import json
import os

import numpy as np
from sklearn.model_selection import KFold


def create_dataset_file(config):
    image_folder_name = 'imagesTr'
    label_folder_name = 'labelsTr'

    dataset_path = config['project']['dataset_path']
    image_path = os.path.join(dataset_path, image_folder_name)
    label_path = os.path.join(dataset_path, label_folder_name)

    assert os.path.exists(image_path), f'image folder not found -> {image_path}'
    assert os.path.exists(label_path), f'label folder not found -> {label_path}'

    images = os.listdir(os.path.join(dataset_path, 'imagesTr'))
    labels = os.listdir(os.path.join(dataset_path, 'labelsTr'))

    assert set(images) == set(labels), f'files differ -> {set(labels).symmetric_difference(set(images))}'

    datalist = {'training': []}
    images = sorted(images)
    for image in images:
        datalist['training'].append(
            {
                'image': os.path.join(dataset_path, image_folder_name, image),
                'label': os.path.join(dataset_path, label_folder_name, image),
            }
        )

    config_path = config['project']['config_path']
    os.makedirs(config_path, exist_ok=True)
    dataset_file_path = os.path.join(config_path, 'dataset.json')
    with open(dataset_file_path, 'w') as f:
        json.dump(datalist, f)
    print(f'dataset file has been created -> {dataset_file_path}')


def create_datalist(config):
    dataset_file_path = os.path.join(config['project']['config_path'], 'dataset.json')

    with open(dataset_file_path, 'r') as f:
        dataset = json.load(f)

    dataset_with_folds = dataset.copy()

    keys = [line['image'] for line in dataset['training']]
    dataset_train_dict = dict(zip(keys, dataset['training']))
    all_keys_sorted = np.sort(keys)

    kfold = KFold(
        n_splits=config['dataset']['total_folds'],
        shuffle=True,
        random_state=config['dataset']['seed'],
    )
    for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
        val_data = []
        train_data = []
        train_keys = np.array(all_keys_sorted)[train_idx]
        test_keys = np.array(all_keys_sorted)[test_idx]
        for key in test_keys:
            val_data.append(dataset_train_dict[key])
        for key in train_keys:
            train_data.append(dataset_train_dict[key])

        dataset_with_folds[f'val_fold_{i}'] = val_data
        dataset_with_folds[f'train_fold_{i}'] = train_data
    del dataset

    with open(dataset_file_path, 'w') as f:
        json.dump(dataset_with_folds, f)
    print(f'dataset file with folds has been created -> {dataset_file_path}')
