# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import tensorflow as tf

from model.predict_model import PredictModel
from utils.my_file import file_list, file_list_recursive

if __name__ == '__main__':
    pm = PredictModel("../save/model/flowers102")
    files=file_list_recursive("/WORK/datasset/102flowers/val/80")
    for i in files:
        result = pm.predict(i, 256, 256)
        for re in result:
            print(tf.argmax(re))
