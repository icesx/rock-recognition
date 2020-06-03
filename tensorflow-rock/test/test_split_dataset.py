# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import unittest
from unittest import TestCase
from dataset.split_dataset import random_dataset


class TestSplitDataset(TestCase):
    def test_split_star(self):
        random_dataset(root="/WORK/datasset/star_imgs",
                       train_root="/WORK/datasset/star_imgs_train",
                       test_root="/WORK/datasset/star_imgs_test")

    def test_split_rock(self):
        random_dataset(root="/WORK/datasset/rock_imgs",
                       train_root="/WORK/datasset/rock_imgs_train",
                       test_root="/WORK/datasset/rock_imgs_test")


if __name__ == '__main__':
    unittest.main()
