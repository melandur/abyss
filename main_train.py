from main_conf import params
from src.data_set.data_set_init_path_scan import DataSetInitPathScan


if __name__ == '__main__':
    ds_init_path_scan = DataSetInitPathScan(data_set_path=params['dataset']['folder_path'],
                                            label_search_tags=params['dataset']['label_search_tags'],
                                            label_file_type=params['dataset']['label_file_type'],
                                            image_search_tags=params['dataset']['image_search_tags'],
                                            image_file_type=params['dataset']['image_file_type'])





    # prepare data

    print(ds_init_path_scan.data_path_store)

    # start train
