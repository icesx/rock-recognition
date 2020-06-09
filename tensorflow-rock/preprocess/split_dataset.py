# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import random

from dataset.image_file import image_labels
from utils.my_file import copy, mk_dir


def random_dataset(root, ratio, train_root, test_root):
    image_infos = image_labels(root)
    print("path 10 images", image_infos[0:10])
    random.shuffle(image_infos)
    paths_len = len(image_infos)
    i = int(paths_len * ratio)
    train = image_infos[:i]
    test = image_infos[i:]
    __copy(train, train_root)
    __copy(test, test_root)


def __copy(srcs, target):
    for image_info in srcs:
        folder = target + "/" + image_info.path.parent.name
        mk_dir(folder)
        copy(image_info.path_str, folder + "/" + image_info.path.name)
