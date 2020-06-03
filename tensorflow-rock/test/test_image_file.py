# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
<<<<<<< HEAD
from dataset.image_file import *
=======
from utils.image_file import *
>>>>>>> c35f7f48f7b3e88259866ff6f548f1ab53f7cbe3
import unittest


class TestImages(unittest.TestCase):
    def test_load_images(self):
        images = images_byte_array("/WORK/datasset/rock_imgs_test", 128, 128)
        print(images[:10])


if __name__ == '__main__':
    unittest.main()
