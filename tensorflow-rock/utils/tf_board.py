# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import datetime
import tensorflow as tf


<<<<<<< HEAD
def tf_board(name):
    log_dir = "../tmp/logs/tb/"+name+"/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
=======
def tf_board():
    log_dir = "../tmp/logs/tb/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
>>>>>>> c35f7f48f7b3e88259866ff6f548f1ab53f7cbe3
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
