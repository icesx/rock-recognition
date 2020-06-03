# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import datetime
import tensorflow as tf


def tf_board(name):
    log_dir = "../tmp/logs/tb/" + name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)