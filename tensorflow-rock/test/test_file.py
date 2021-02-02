# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import unittest
from unittest import TestCase

from utils.my_file import *


class TestFile(TestCase):
    def test_file_list(self):
        ls = file_list("/WORK/datasset/mnist/test/0")
        for f in ls:
            print(f)


if __name__ == '__main__':
    unittest.main()
    ls = file_list("/WORK/datasset/mnist/test/0")
    for f in ls:
        print(f)