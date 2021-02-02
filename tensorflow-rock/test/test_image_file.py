# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from dataset.image_file import image_byte_array
import unittest


class TestImages(unittest.TestCase):
    def test_load_images(self):
        images = image_byte_array("/WORK/datasset/rock_imgs_test", 128, 128)
        print(images[:10])


if __name__ == '__main__':
    unittest.main()
