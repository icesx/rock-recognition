# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import tensorflow as tf

from dataset.dataset_creator import CustomDatasetCreator
from model.predict_model import PredictModel
from utils.my_file import file_current

IMG_X = 130
IMG_Y = 155


def predict_files():
    pm = PredictModel("../save/model/star")
    for i in file_current("/WORK/datasset/star_imgs_test/liuyifei/"):
        result = pm.predict(i, IMG_X, IMG_Y)
        for re in result:
            print(tf.argmax(re))


def evaluate():
    pm = PredictModel("../save/model/star")
    ds = CustomDatasetCreator("/WORK/datasset/star_imgs_test/",
                              image_x=IMG_X,
                              image_y=IMG_Y).get()
    result = pm.evaluate(ds)
    print(result)


if __name__ == '__main__':
    # predict_files()
    evaluate()
