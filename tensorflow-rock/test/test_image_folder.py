# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
# 2/23/21
from pathlib import Path

import tensorflow as tf

from dataset.image_file import image_byte_array


def _decode_image(image_path):
    return image_byte_array(image_path, 224, 224)


def _decode_labels(image_path):
    return tf.string(Path(image_path).parent.name)


def file_map(image_path):
    print("xxx", image_path)
    return (
        _decode_image(image_path),
        _decode_labels(image_path)
    )


if __name__ == '__main__':
    dataset = tf.data.Dataset.list_files("/WORK/datasset/102flowers/train/**/*.jpg");
    for i in dataset.take(10):
        print(i)

    dataset.map(map_func=file_map)