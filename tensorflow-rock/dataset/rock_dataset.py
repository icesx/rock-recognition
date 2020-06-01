import tensorflow as tf
import pathlib
import random
from utils.image_file import *
from utils.plot import *
from utils.my_file import file_resave


class RockDataset:
    def __init__(self, image_x=64, image_y=64):
        self.image_x = image_x
        self.image_y = image_y

    def load(self, root, batch):
        return self.__create_dataset(root, batch)

    def __load_and_preprocess_from_path_label(self, path, lable):
        images = image_byte_array(path, self.image_x, self.image_y)
        return images, lable

    def __create_dataset(self, root, batch):
        image_label=image_labels(root)
        ds = tf.data.Dataset.from_tensor_slices((image_label.paths, image_label.label_idx))
        ds = ds.map(self.__load_and_preprocess_from_path_label)
        ds = ds.cache()
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1024))
        return ds.batch(batch), image_label


if __name__ == '__main__':
    ds = RockDataset().load('/WORK/datasset/rock_imgs_train2', 15)
    for element in ds:
        print(element)
