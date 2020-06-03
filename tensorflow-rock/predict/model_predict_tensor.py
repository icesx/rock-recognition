# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from tensorflow import keras
import tensorflow as tf
from dataset.image_file import image_byte_array
import numpy as np


def load_model(export_dir):
    print("load model from ", export_dir)
    return keras.models.load_model(export_dir)


def predict(model, img):
    result = model.predict(np.expand_dims(image_byte_array(img, 128, 128), axis=0))
    for r in result:
        print(tf.argmax(r), r)


if __name__ == '__main__':
    from utils.gpu import gpu_init
    gpu_init(6000)
    model = load_model("../save/rock")
    for i in range(1, 20):
        predict(model=model, img="/WORK/datasset/star_imgs/pengyuyan/{0}.png".format(i))
