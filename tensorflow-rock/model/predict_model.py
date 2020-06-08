# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import numpy as np
from tensorflow import keras
import tensorflow as tf


class PredictModel:
    def __init__(self, export_dir):
        from utils.tf_gpu import gpu_init
        gpu_init(6000)
        print("load model from ", export_dir)
        self.__model = keras.models.load_model(export_dir)
        self.__model_predict = keras.Sequential([self.__model,
                                                 keras.layers.Softmax()])

    def image_byte_array(self, path, image_x, image_y):
        img = tf.keras.preprocessing.image.load_img(path, target_size=(image_x, image_y))
        return img

    def evaluate(self, dataset):
        return self.__model.evaluate(dataset, steps=10)

    def predict(self, img_path, image_x, image_y):
        dims = np.expand_dims(self.image_byte_array(img_path, image_x, image_y), axis=0)
        return self.__model_predict.predict(dims / 255.0)
