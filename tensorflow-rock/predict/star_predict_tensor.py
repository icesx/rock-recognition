# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import tensorflow as tf

from model.predict_model import PredictModel

if __name__ == '__main__':
    pm = PredictModel("../save/model/star")
    for i in range(1, 20):
        result = pm.predict("/WORK/datasset/star_imgs/pengyuyan/{0}.png".format(i), 128, 128)
        for re in result:
            print(re, tf.argmax(re))
