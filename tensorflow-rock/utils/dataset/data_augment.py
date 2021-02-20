# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
# 2/19/21
import tensorflow as tf

from utils.plot import plot_shows_2


def rot90(image, label):
    return tf.image.rot90(image), label


def random_brightness(image, label):
    return tf.image.random_brightness(image, max_delta=0.4), label


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


class ImageAugment:
    def __init__(self, path):
        self.__path = path
        self.__image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)  # 彩色图像为3个channel
        self.__images = []

    def __rot90(self, k=1):
        __image = tf.image.rot90(self.__image, k)
        self.__images.append(__image)
        return self

    def rot90(self):
        return self.__rot90()

    def rot270(self):
        return self.__rot90(k=3)

    def flip_up_down(self):
        __image = tf.image.flip_up_down(self.__image)
        self.__images.append(__image)
        return self

    def flip_left_right(self):
        __image = tf.image.flip_left_right(self.__image)
        self.__images.append(__image)
        return self

    def central_crop(self):
        __image = tf.image.central_crop(self.__image, central_fraction=0.8)
        self.__images.append(__image)
        return self

    def random_brightness(self):
        __image = tf.image.random_brightness(self.__image, max_delta=0.4)
        self.__images.append(__image)
        return self

    def images(self):
        return self.__images


if __name__ == '__main__':
    images = ImageAugment(
        "/WORK/datasset/flower_photos/train/daisy/99306615_739eb94b9e_m.jpg") \
        .flip_up_down() \
        .flip_left_right() \
        .central_crop() \
        .random_brightness() \
        .rot90() \
        .rot270() \
        .images()
    for i in images:
        print(i)
    plot_shows_2(images)
