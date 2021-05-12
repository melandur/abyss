
from monai import transforms as tf


class ConcatenateImages(tf.MapTransform):
    """MONAI expects a simple dict. But we like our dicts nested, here we go"""

    def __call__(self, data, dim=0, add_channel=True):
        d = dict(data)
        self.dim = dim
        self.add_channel = add_channel

        def load_image_sub_dict(sub_dict, keys):
            """Load image and add channel to the first dimension"""
            if self.add_channel:
                load_image = tf.Compose([
                    tf.LoadImaged(keys=keys, image_only=True),
                    tf.AddChanneld(keys=keys)])
            else:
                load_image = tf.Compose([tf.LoadImaged(keys=keys, image_only=True)])
            return load_image(sub_dict)

        def concatenate_images_sub_dict(sub_dict, keys):
            concatenate_images = tf.Compose([tf.ConcatItemsd(keys=keys, name='concat_output', dim=self.dim)])
            return concatenate_images(sub_dict)

        for key in self.keys:
            if 'image' in key:
                for case_name in d[key].keys():
                    tmp_sub_dict = {}
                    for image_name in d[key][case_name].keys():  # create sub dict for a single case
                        tmp_sub_dict[image_name] = d[key][case_name][image_name]

                    tmp_sub_images = {'concat_output': None}  # adds key to dict to store concat result later on
                    for image_name in [*tmp_sub_dict.keys()]:  # load image from each image path in sub dict
                        tmp = load_image_sub_dict(tmp_sub_dict, image_name)
                        tmp_sub_images[image_name] = tmp[image_name]

                    # assign concat results to input by overwriting the previous sub dicts
                    d['image'][case_name] = concatenate_images_sub_dict(tmp_sub_images, [*tmp_sub_dict.keys()])
        return d
