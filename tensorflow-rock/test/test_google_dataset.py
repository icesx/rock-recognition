# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow as tf

if __name__ == '__main__':
    train, test = tfds.load('tf_flowers', split=['train', 'test'])
