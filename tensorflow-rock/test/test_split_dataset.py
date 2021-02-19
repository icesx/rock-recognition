# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from utils.dataset.split_dataset import random_dataset


def test_split_star():
    random_dataset(root="/WORK/datasset/star_imgs/all",
                   train_ratio=0.9,
                   train_root="/WORK/datasset/star_imgs/train",
                   test_root="/WORK/datasset/star_imgs/val")


def test_split_102():
    random_dataset(root="/WORK/datasset/102flowers/all",
                   train_ratio=0.9,
                   train_root="/WORK/datasset/102flowers/train",
                   test_root="/WORK/datasset/102flowers/val")


def test_split_rock():
    random_dataset(root="/WORK/datasset/rock_imgs",
                   train_root="/WORK/datasset/rock_imgs_train",
                   test_root="/WORK/datasset/rock_imgs_test")


def save_tf_flowers():
    random_dataset(train_ratio=0.9,
                   root="/WORK/datasset/flower_photos/all",
                   train_root="/WORK/datasset/flower_photos/train",
                   test_root="/WORK/datasset/flower_photos/val")


if __name__ == '__main__':
    save_tf_flowers()
