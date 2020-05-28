# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

from tensorflow import keras


def load_module(export_dir):
    print("load model from ", export_dir)
    return keras.models.load_model(export_dir)


if __name__ == '__main__':
    module = load_module("../tmp/save/rock")
    print(module)
    # prediced = module.predict(test_images)
    # print(test_labels[0], tf.argmax(prediced[0]))
