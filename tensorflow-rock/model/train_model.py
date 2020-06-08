# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

import tensorflow as tf
from tensorflow import keras

from model.model_creater import BaseModelOperate


class RockModel(BaseModelOperate):
    def __init__(self, train_image_root, test_image_root, image_x, image_y):
        BaseModelOperate.__init__(self, train_image_root, test_image_root, image_x, image_y)

    def _provide_model(self, image_x, image_y):
        model = keras.Sequential([
            keras.layers.Conv2D(120, (2, 2), activation="relu", input_shape=(image_x, image_y, 3)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (2, 2), activation="relu"),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(32, (2, 2), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(167)
        ])
        return model


class StarModel(BaseModelOperate):
    def __init__(self, train_image_root, test_image_root, image_x, image_y):
        BaseModelOperate.__init__(self, train_image_root, test_image_root, image_x, image_y)

    def _provide_model(self, image_x, image_y):
        model = keras.Sequential([
            keras.layers.Conv2D(10, (5, 5), activation=tf.nn.relu, input_shape=(image_x, image_y, 3),
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.001)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(20, (3, 3), activation=tf.nn.relu, input_shape=(image_x, image_y, 3),
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                                bias_regularizer=tf.keras.regularizers.l2(l=0.001)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.001)),
            keras.layers.Dense(10)
        ])
        return model


class FlowerModel(BaseModelOperate):
    def __init__(self, train_image_root, test_image_root, image_x, image_y):
        BaseModelOperate.__init__(self, train_image_root, test_image_root, image_x, image_y)

    def _provide_model(self, image_x, image_y):
        model = keras.Sequential([
            keras.layers.Conv2D(64, (3, 3),
                                activation=tf.nn.relu,
                                input_shape=(image_x, image_y, 3)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3),
                                activation=tf.nn.relu),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512,
                               activation=tf.nn.relu,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.05),
                               bias_regularizer=tf.keras.regularizers.l2(l=0.01)),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(128,
                               activation=tf.nn.relu,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.03),
                               bias_regularizer=tf.keras.regularizers.l2(l=0.05)),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(5)
        ])
        return model


class Flower102Model(BaseModelOperate):
    def __init__(self, image_x, image_y):
        BaseModelOperate.__init__(self,image_x, image_y)

    def _provide_model(self, image_x, image_y):
        model = keras.Sequential([
            keras.layers.Conv2D(64, (3, 3),
                                activation=tf.nn.relu,
                                input_shape=(image_x, image_y, 3)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(256,
                               activation=tf.nn.relu,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                               bias_regularizer=tf.keras.regularizers.l2(l=0.001)),
            keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(128,
                               activation=tf.nn.relu,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                               bias_regularizer=tf.keras.regularizers.l2(l=0.001)),
            keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(102)
        ])
        return model
