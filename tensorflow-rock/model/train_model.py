# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

import tensorflow as tf
from dataset.create_dataset import DatasetCreator
from tensorflow import keras

from utils.my_file import over_write
from utils.tf_board import tf_board
from dataset.image_file import ALL_LABELS


class BaseModelOperate:
    def __init__(self, image_root, image_x, image_y, module_name=None):
        from utils.tf_gpu import gpu_init
        gpu_init(6000)
        self.__model = None
        self.__ds = None
        __module_name = module_name if module_name is not None else self.__class__.__name__
        self.__label_file = "../save/lables_{0}.csv".format(__module_name)
        self.__image_x = image_x
        self.__image_y = image_y
        self.__image_root = image_root

    def load(self, batch=10):
        creator = DatasetCreator(image_x=self.__image_x, image_y=self.__image_y)
        self.__ds = creator.load(
            self.__image_root).repeat().batch(batch)
        self.__write_labels()
        return self

    def __write_labels(self):
        with open(self.__label_file, 'w') as file:
            for label in ALL_LABELS.items():
                file.write(label[0] + "," + str(label[1].label_idx) + "\r\n")

    def _create(self, image_x, image_y):
        return None

    def train(self, steps_per_epoch, epochs=100):
        self.__model = self._create(self.__image_x, self.__image_y)
        if self.__model is None:
            print("please overrider _train()")
        else:
            self.compile(self.__model)
            self.__model.summary()
            self.fit(self.__ds, self.__model, epochs, steps_per_epoch)
            return self

    def save(self, save_dir):
        tf.saved_model.save(self.__model, export_dir=save_dir)

    def fit(self, dataset, model, epochs, steps_per_epoch):
        model.fit(dataset,
                  epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  callbacks=[tf_board(self.__class__.__name__)])

    def compile(self, model):
        model.compile(optimizer="adam",
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])


class RockModel(BaseModelOperate):
    def __init__(self, image_root, image_x, image_y):
        BaseModelOperate.__init__(self, image_root, image_x, image_y)

    def _create(self, image_x, image_y):
        model = keras.Sequential([
            keras.layers.Conv2D(120, (2, 2), activation="relu", input_shape=(image_x, image_y, 3)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (2, 2), activation="relu"),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (2, 2), activation="relu"),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (2, 2), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(167)
        ])
        return model


class StarModel(BaseModelOperate):
    def __init__(self, image_root, image_x, image_y):
        BaseModelOperate.__init__(self, image_root, image_x, image_y)

    def _create(self, image_x, image_y):
        model = keras.Sequential([
            keras.layers.Conv2D(120, (2, 2), activation="relu", input_shape=(image_x, image_y, 3)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (2, 2), activation="relu"),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (2, 2), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(120, activation=tf.nn.relu),
            keras.layers.Dense(10)
        ])
        return model


class Mnist(BaseModelOperate):
    def __init__(self, image_root, image_x=28, image_y=28, module_name="mnist"):
        BaseModelOperate.__init__(self, image_root, image_x, image_y, module_name)

    def _create(self, image_x, image_y):
        model = keras.Sequential([
            keras.layers.Conv2D(120, (2, 2), activation="relu", input_shape=(image_x, image_y, 3)),
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(120, activation=tf.nn.relu),
            keras.layers.Dense(10)
        ])
        return model


class FashionMnist(Mnist):
    def __init__(self, image_root, image_x=28, image_y=28):
        Mnist.__init__(self, image_root, image_x, image_y, "fashion_mnist")
