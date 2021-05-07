import os
from src.utils import NestedDefaultDict, check_instance


class DataSetInitPathScan:
    """Creates a nested dictionary, which holds keys:case_names, values: label and img paths"""

    def __init__(self, data_set_path, label_search_tags, label_file_type, image_search_tags, image_file_type):
        self.data_set_path = data_set_path
        self.label_search_tags = check_instance(label_search_tags)
        self.label_file_type = check_instance(label_file_type)
        self.image_search_tags = check_instance(image_search_tags)
        self.image_file_type = check_instance(image_file_type)
        self.data_path_store = NestedDefaultDict()

        self.scan_folder()

    @staticmethod
    def get_case_name(file_name):
        """Extracts specific case name from file name"""
        # TODO: Depends heavily on the naming of your data set
        case_name = '_'.join(file_name.split('_')[:-1])
        # print(case_name)
        return case_name

    def check_file_search_tag_label(self, file_name):
        """True if label search tag is in file name"""
        check_search_tag = False
        if [x for x in self.label_search_tags if x in file_name]:
            check_search_tag = True
        return check_search_tag

    def check_file_type_label(self, file_name):
        """True if label file ends with defined file type"""
        check_file_type = False
        if [x for x in self.label_file_type if file_name.endswith(x)]:
            check_file_type = True
        return check_file_type

    def check_file_search_tag_image(self, file_name):
        """True if image search tag is in file name"""
        check_search_tag = False
        if [x for x in self.image_search_tags if x in file_name]:
            check_search_tag = True
        return check_search_tag

    def check_file_type_image(self, file_name):
        """True if image file ends with defined file type"""
        check_file_type = False
        if [x for x in self.image_file_type if file_name.endswith(x)]:
            check_file_type = True
        return check_file_type

    def get_file_search_tag_image(self, file_name):
        """Returns the found search tag for a certain file name"""
        return [x for x in self.image_search_tags if x in file_name][0]

    def scan_folder(self):
        """Walk through the data set folder and assigns file paths to the nested dict"""
        for root, dirs, files in os.walk(self.data_set_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    if self.check_file_search_tag_label(file) and self.check_file_type_label(file):
                        self.data_path_store[self.get_case_name(file)]['label'] = file_path
                    if self.check_file_search_tag_image(file) and self.check_file_type_image(file):
                        found_tag = self.get_file_search_tag_image(file)
                        self.data_path_store[self.get_case_name(file)]['image'][found_tag] = file_path


if __name__ == '__main__':
    dl = DataSetInitPathScan(
        data_set_path=r'C:\Users\melandur\Desktop\MICCAI_BraTS_2019_Data_Training\MICCAI_BraTS_2019_Data_Training\HGG',
        label_search_tags=['seg'],
        label_file_type=['.nii.gz'],
        image_search_tags=['t1', 't1ce', 'flair', 't2'],
        image_file_type=['.nii.gz'])
    dl.scan_folder()
    print(dl.data_path_store)
