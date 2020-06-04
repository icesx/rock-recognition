# coding:utf-8
# Copyright (C)
# Author: I
import tensorflow as tf
import numpy as np
from tensorflow import keras

from utils.my_file import file_list


def augment(path, image_x, image_y):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image_1 = tf.image.resize_with_crop_or_pad(image, image_x - int(image_x * 0.01), image_y - int(image_y * -0.01))
    save(path + "_crop", image_1)
    image_2 = tf.image.adjust_contrast(image, 0.5)
    save(path + "_adjust_contrast", image_2)
    image_3 = tf.image.random_brightness(image, max_delta=0.5)
    save(path + "_random_brightness", image_3)
    image_4 = tf.image.random_flip_left_right(image,2)
    save(path + "_random_flip_1", image_4)
    image_5 = tf.image.random_flip_up_down(image,3)
    save(path + "_random_flip_2", image_5)
    return image


def save(path, image):
    path = path + "_AUG_" + ".jpg"
    print("save to ", path)
    keras.preprocessing.image.save_img(path, image)


if __name__ == '__main__':
    listf = file_list("/WORK/datasset/star_imgs_train/")
    for f in listf:
        if "_AUG_" not in f:
            augment(f, 128, 128)
