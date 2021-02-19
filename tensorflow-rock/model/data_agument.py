# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
# 2/19/21
import tensorflow as tf
from tensorflow.keras import layers


def resize_rescale(image_x, image_y):
    return tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(image_x, image_y, input_shape=(image_x, image_y, 3)),
        layers.experimental.preprocessing.Rescaling(1. / 255)
    ])


def flip_rotation():
    return tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2)
    ])


def rot90(image, label):
    return tf.image.rot90(image), label


def random_brightness(image, label):
    return tf.image.random_brightness(image,max_delta=0.4), label


def central_crop(image, label):
    return tf.image.central_crop(image, central_fraction=0.9), label


def flip_up_down(image, label):
    image = tf.image.flip_up_down(image)
    return image, label


def flip_left_right(image, label):
    flip_image = tf.image.flip_left_right(image)
    return flip_image, label


def rgb_to_grayscale(image, label):
    return tf.image.rgb_to_grayscale(image), label
