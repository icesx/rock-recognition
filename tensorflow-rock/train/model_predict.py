# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

from tensorflow import keras
import tensorflow as tf
import numpy as np

def load_model(export_dir):
    print("load model from ", export_dir)
    return keras.models.load_model(export_dir)


if __name__ == '__main__':
    model = load_model("../tmp/save/rock")
    print(model)
    # test_images=
    # predicted = model.predict(test_images)
    # for i in predicted:
    #     print(i)
