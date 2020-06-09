# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import tensorflow as tf

from dataset.dataset_creator import CustomDatasetCreator


class DatasetGroup:
    def __init__(self):
        pass

    def validation(self, batch) -> tf.data.Dataset:
        pass

    def train(self, batch) -> tf.data.Dataset:
        pass


class CustomDatasetGroup(DatasetGroup):
    def __init__(self, train_root, validation_root, image_y, image_x):
        DatasetGroup.__init__(self)
        self.__image_x = image_x
        self.__image_y = image_y
        self.__validation = CustomDatasetCreator(validation_root, image_y, image_x)
        self.__train = CustomDatasetCreator(train_root, image_y, image_x)

    @property
    def image_x(self):
        return self.__image_x;

    @property
    def image_y(self):
        return self.__image_y;

    def validation(self, batch) -> tf.data.Dataset:
        return self.__validation.batch(batch).get()

    def train(self, batch) -> tf.data.Dataset:
        return self.__train.batch(batch).get()


class TfDatasetGroup(DatasetGroup):
    def __init__(self, name):
        DatasetGroup.__init__(self)

    def validation(self, batch) -> tf.data.Dataset:
        pass

    def train(self, batch) -> tf.data.Dataset:
        pass
