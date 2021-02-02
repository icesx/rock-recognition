# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

import tensorflow as tf

from model.predict_model import PredictModel
from utils.my_file import file_list

if __name__ == '__main__':
    pm = PredictModel("../save/model/fashion_mnist")
    for i in file_list("/WORK/datasset/fashion_mnist/test/Bag"):
        result = pm.predict(i, 28, 28)
        for re in result:
            print(tf.argmax(re))
