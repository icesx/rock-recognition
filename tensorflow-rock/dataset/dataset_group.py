# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import tensorflow as tf

from dataset.create_dataset import DatasetCreator
from dataset.image_file import ALL_LABELS


class DatasetGroup:
    def __init__(self):
        pass

    def validation(self, batch) -> tf.data.Dataset:
        pass

    def train(self, batch) -> tf.data.Dataset:
        pass

    def write_labels(self, label_file_name):
        pass


class CustomDatasetGroup(DatasetGroup):
    def __init__(self, train_root, validation_root, image_y, image_x):
        from utils.tf_gpu import gpu_init

        gpu_init(6000)
        self.image_x = image_x
        self.image_y = image_y
        DatasetGroup.__init__(self)
        self.__validation = DatasetCreator(validation_root, image_y, image_x)
        self.__train = DatasetCreator(train_root, image_y, image_x)

    def validation(self, batch) -> tf.data.Dataset:
        return self.__validation.batch(batch).get()

    def train(self, batch) -> tf.data.Dataset:
        return self.__train.batch(batch).get()

    def write_labels(self, label_file_name):
        __label_file = "../save/labels_{0}.csv".format(self.__class__.__name__)
        with open(label_file_name, 'w') as file:
            for label in ALL_LABELS.items():
                file.write(label[0] + "," + str(label[1].label_idx) + "\r\n")


import tensorflow_datasets as tfds


class TfDatasetGroup(DatasetGroup):
    def __init__(self, name):
        DatasetGroup.__init__(self)

    def validation(self, batch) -> tf.data.Dataset:
        pass

    def train(self, batch) -> tf.data.Dataset:
        pass
