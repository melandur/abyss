import sys

params = {

    'logger': {'level': 'INFO'},  # 'TRACE', 'DEBUG', 'INFO', 'WARN'

    'project': {'name': 'BratsExp1',
                'dataset_store_path': r'C:\Users\melandur\Desktop\mo',
                'result_store_path': r'C:\Users\melandur\Desktop\mo\logs'},

    'dataset': {
        'folder_path': r'C:\Users\melandur\Desktop\MICCAI_BraTS_2019_Data_Training\MICCAI_BraTS_2019_Data_Training\HGG',
        'label_search_tags': ['seg'],
        'label_file_type': ['.nii.gz'],
        'image_search_tags': ['t1', 't1ce', 'flair', 't2'],
        'image_file_type': ['.nii.gz'],

        'pull_dataset': 'DecathlonDataset',  # 'MedNISTDataset', 'DecathlonDataset', 'CrossValidation'
        'challenge': 'Task01_BrainTumour',  # only need for decathlon:   'Task01_BrainTumour', 'Task02_Heart', 'Task03_Liver0', 'Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon'
        'seed': 42,
        'val_frac': 0.2,
        'test_frac': 0.2,
        'use_cache': False,  # goes heavy on memory
        'cache_max': sys.maxsize,
        'cache_rate': 0.0,  # set 0 to reduce memory consumption
        'num_workers': 4
    },

    'pre_processing': {},
    'post_processing': {},

    'augmentation': {},

    'training': {
        'seed': 42,
        'epochs': 30,  # tbd
        'trained_epochs': None,
        'batch_size': 1,  # tbd
        'optimizer': 'Adam',  # Adam, SGD
        'learning_rate': 1e-3,  # tbd
        'betas': (0.9, 0.999),  # tbd
        'eps': 1e-8,
        'weight_decay': 1e-5,  # tbd
        'amsgrad': True,
        'dropout': 0.5,  # tbd
        'criterion': ['MSE_mean'],
        'num_workers': 8,
        'n_classes': 3},
}
