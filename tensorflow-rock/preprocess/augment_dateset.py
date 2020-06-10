# coding:utf-8
# Copyright (C)
# Author: I
import tensorflow as tf
import random
from utils.my_file import file_list
import matplotlib as plot


def augment(path, image_x, image_y):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image_1 = tf.image.resize_with_crop_or_pad(image, image_x - int(image_x * 0.01), image_y - int(image_y * -0.01))
    __save_numpy_image(path + "_crop", image_1)
    image_2 = tf.image.adjust_contrast(image, 0.5)
    __save_numpy_image(path + "_adjust_contrast", image_2)
    image_3 = tf.image.random_brightness(image, max_delta=0.5)
    __save_numpy_image(path + "_random_brightness", image_3)
    image_4 = tf.image.flip_left_right(image)
    __save_numpy_image(path + "_flip_left_right", image_4)
    image_5 = tf.image.flip_up_down(image)
    __save_numpy_image(path + "_flip_up_down", image_5)
    image_6 = tf.image.rot90(image_5, k=random.randint(1, 3))
    __save_numpy_image(path + "_rot90", image_6)
    return image


def __save_numpy_image(path, image):
    path = path + "_AUG_" + ".jpg"
    print("AUG save to ", path)
    plot.image.imsave(path, image.numpy())


if __name__ == '__main__':
    listf = file_list("/WORK/datasset/102flowers_train")
    for f in listf:
        if "_AUG_" not in f:
            augment(f, 224, 224)
