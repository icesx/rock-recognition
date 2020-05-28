# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from dataset.image_file import image_labels
import random
from utils.my_file import copy


def random_dataset(root, train_root, test_root):
    paths = image_labels(root).paths
    print("path 10 images", paths[0:10])
    random.shuffle(paths)
    paths_len = len(paths)
    train = paths[:int(paths_len * 0.8)]
    test = paths[int(paths_len * 0.2):]
    __copy(train, train_root)
    __copy(test, test_root)


def __copy(srcs, target):
    for f in srcs:
        copy(f, target)
