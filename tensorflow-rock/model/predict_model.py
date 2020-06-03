# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import numpy as np
from tensorflow import keras

from dataset.image_file import image_byte_array


class PredictModel:
    def __init__(self, export_dir):
        from utils.tf_gpu import gpu_init
        gpu_init(6000)
        print("load model from ", export_dir)
        self.__model = keras.models.load_model(export_dir)

    def predict(self, img_path, image_x, image_y):
        return self.__model.predict(np.expand_dims(image_byte_array(img_path, image_x, image_y), axis=0))
