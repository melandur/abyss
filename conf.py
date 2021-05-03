params = {
    'user': {'dataset_store_path': r'C:\Users\melandur\Desktop\mo'},

    'data': {'dataset': 'DecathlonDataset',  # 'MedNISTDataset', 'DecathlonDataset', 'CrossValidation', 'CustomDataset'
             'challenge': 'Task01_BrainTumour',  # only need for decathlon:   'Task01_BrainTumour', 'Task02_Heart', 'Task03_Liver0', 'Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon'
             'num_workers': 4,
             'use_cache': False,  # goes heavy on memory
             'cache_rate': 1.0,
             'seed': 0,  # 0
             'val_frac': 0.2,
             'test_frac': 0.2},

    # TODO: Expand for data augmentation

    'training': {'epochs': 30,  # tbd
                 'trained_epochs': None,
                 'batch_size': 30,  # tbd
                 'optimizer': 'Adam',
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
