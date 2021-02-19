# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
# 2/19/21
import tensorflow as tf
from tensorflow.keras import layers


def resize_rescale(image_x, image_y):
    return tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(image_x, image_y,input_shape=(image_x, image_y, 3)),
        layers.experimental.preprocessing.Rescaling(1. / 255)
    ])


def flip_rotation():
    return tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2)
    ])
