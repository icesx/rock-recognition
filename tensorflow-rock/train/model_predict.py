# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

from tensorflow import keras
import tensorflow as tf
import numpy as np
from utils.image_file import *
from utils.plot import *


def load_model(export_dir):
    print("load model from ", export_dir)
    return keras.models.load_model(export_dir)


if __name__ == '__main__':
    model = load_model("../tmp/save/rock")
    print(model)
    test_images = images_byte_array("/WORK/datasset/rock_imgs_test", 128, 128)
    print("test images:",test_images[:10])
    # plot_show(test_images)
    # predicted = model.predict(test_images)
    # for i in predicted:
    #     print(i)
