# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
<<<<<<< HEAD
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
=======
import random

from dataset.image_file import image_labels
from utils.my_file import copy, mk_dir


def random_dataset(root, train_root, test_root):
    image_infos = image_labels(root)
    print("path 10 images", image_infos[0:10])
    random.shuffle(image_infos)
    paths_len = len(image_infos)
    i = int(paths_len * 0.8)
    train = image_infos[:i]
    test = image_infos[i:]
>>>>>>> 410dda6... remove model step-1
    __copy(train, train_root)
    __copy(test, test_root)


def __copy(srcs, target):
<<<<<<< HEAD
    for f in srcs:
        copy(f, target)
=======
    for image_info in srcs:
        folder = target + "/" + image_info.path.parent.name
        mk_dir(folder)
        copy(image_info.path_str, folder + "/" + image_info.path.name)
>>>>>>> 410dda6... remove model step-1
