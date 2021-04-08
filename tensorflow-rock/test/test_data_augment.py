# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
# 2/19/21
from utils.dataset.data_augment import ImageAugment
from utils.my_file import file_list_recursive


def __do_augment(root):
    for file in file_list_recursive(root, "*.jpg"):
        ImageAugment.auto(file).save()


def augment_flowers():
    __do_augment("/WORK/datasset/flower_photos/train")


def augment_102flowers():
    __do_augment("/WORK/datasset/102flowers/train")


def augment_star():
    __do_augment("/WORK/datasset/star_imgs/train")


if __name__ == '__main__':
    augment_star()
