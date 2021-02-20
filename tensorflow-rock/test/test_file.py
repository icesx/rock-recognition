# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import unittest
from unittest import TestCase

from utils.my_file import *


class TestFile(TestCase):
    def test_file_list(self):
        for f in file_list("/WORK/datasset/mnist/test"):
            print(f)

    def test0_file_list_recursive(self):
        for f in file_list_recursive("/WORK/datasset/mnist/test"):
            print(f)


if __name__ == '__main__':
    unittest.main()
