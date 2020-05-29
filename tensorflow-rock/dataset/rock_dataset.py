import tensorflow as tf
import pathlib
import random
from utils.image_file import *


class RockDataset:
    def __init__(self, image_x=64, image_y=64):
        self.image_x = image_x
        self.image_y = image_y

    def load(self, root):
        return self.__create_dataset(root)

    def __load_and_preprocess_from_path_label(self, path, lable):
        images = load_image(path, self.image_x, self.image_y)
        return normal_image(images), lable

    def __create_dataset(self, data_root_orig):
        all_image_paths, all_image_labels = image_labels(data_root_orig)
        ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
        ds = ds.map(self.__load_and_preprocess_from_path_label)
        ds = ds.cache()
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1024))
        return ds.batch(8)


if __name__ == '__main__':
    rd = RockDataset()
    ds = rd.load()
    for element in ds:
        print(element)
