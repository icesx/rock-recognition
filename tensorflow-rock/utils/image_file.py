# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import pathlib
import random
import tensorflow as tf


def image_paths(data_root):
    all_image_paths = list(data_root.glob('*/*'))
    print(all_image_paths)
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    print("First 10 image_paths: ", all_image_paths[:10])
    return all_image_paths


def image_label_index(data_root):
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    print('label_names: ', label_names)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print('label_to_index: ', label_to_index)
    return label_to_index


def image_labels(data_root_orig):
    data_root = pathlib.Path(data_root_orig)
    all_image_paths = image_paths(data_root)
    label_to_index = image_label_index(data_root)
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
    print("First 10 labels indices: ", all_image_labels[:10])
    return all_image_paths, all_image_labels


def load_images(path, image_x, image_y):
    pass


def load_image(path, image_x, image_y):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_x, image_y])
    image /= 255.0
    return image


def normal_image(image):
    image /= 255.0
    return image
