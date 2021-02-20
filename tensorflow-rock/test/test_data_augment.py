# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
# 2/19/21
from utils.dataset.data_augment import ImageAugment
from utils.my_file import file_list


def __do_augment(root):
    for file in file_list(root):
        ImageAugment.auto(file).save()


def augment_flowers():
    __do_augment("/WORK/datasset/flower_photos/train")


if __name__ == '__main__':
    augment_flowers()
