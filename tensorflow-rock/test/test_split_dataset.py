# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import unittest
from unittest import TestCase
from preprocess.split_dataset import random_dataset


class TestSplitDataset(TestCase):
    def test_split_star(self):
        random_dataset(root="/WORK/datasset/star_imgs",
                       ratio=0.8,
                       train_root="/WORK/datasset/star_imgs_train",
                       test_root="/WORK/datasset/star_imgs_test")

    def test_split_rock(self):
        random_dataset(root="/WORK/datasset/rock_imgs",
                       ratio=0.8,
                       train_root="/WORK/datasset/rock_imgs_train",
                       test_root="/WORK/datasset/rock_imgs_test")

    def test_split_flower(self):
        random_dataset(root="/WORK/datasset/flower_photos",
                       ratio=0.8,
                       train_root="/WORK/datasset/flower_photos_train",
                       test_root="/WORK/datasset/flower_photos_test")

    def test_split_102flower(self):
        random_dataset(root="/WORK/datasset/102flowers",
                       ratio=0.8,
                       train_root="/WORK/datasset/102flowers_train",
                       test_root="/WORK/datasset/102flowers_test")


if __name__ == '__main__':
    unittest.main()
