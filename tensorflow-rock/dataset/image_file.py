# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import pathlib
import random
import tensorflow as tf
import numpy as np


class ImageLabels:
    def __init__(self, paths, label_idx, label_name, label_name_idx):
        self.__paths = paths
        self.__label_idx = label_idx
        self.__label_name = label_name
        self.__label_name_idx = label_name_idx
        self.__idx_label_name = {v: k for k, v in label_name_idx.items()}

    @property
    def paths(self):
        return self.__paths

    @property
    def label_idx(self):
        return self.__label_idx

    @property
    def label_name(self):
        return self.__label_name

    @property
    def label_name_idx(self):
        return self.__label_name_idx

    @property
    def idx_name_label(self):
        return self.__idx_label_name


def image_paths(root_path):
    root_path = pathlib.Path(root_path)
    all_image_paths = list(root_path.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    print("First 10 image_paths: ", all_image_paths[:10])
    return all_image_paths


def image_label_index(root_path):
    label_names = sorted(item.name for item in root_path.glob('*/') if item.is_dir())
    print('label_names: ', label_names)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print('label_to_index: ', label_to_index)
    return label_to_index, label_names


def image_labels(root: object) -> ImageLabels:
    root_path = pathlib.Path(root)
    all_image_paths = image_paths(root_path)
    label_name_to_index, lable_names = image_label_index(root_path)
    all_image_labels_id = [label_name_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
    print("First 10 images: ", all_image_paths[:10])
    print("First 10 labels indices: ", all_image_labels_id[:10])
    return ImageLabels(all_image_paths, all_image_labels_id, lable_names, label_name_to_index)


def images_byte_array(root_path, image_x, image_y):
    all_image_paths, all_image_labels, label_to_index = image_labels(pathlib.Path(root_path))
    return np.array([image_byte_array(path, image_x, image_y) for path in all_image_paths]), np.array(all_image_labels)


def image_byte_array(path, image_x, image_y):
    print(path)
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_x, image_y])
    image /= 255.0
    return image
