# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import tensorflow as tf

from dataset.create_dataset import DatasetCreator
from model.predict_model import PredictModel
from utils.my_file import file_current

IMG_X = 128
IMG_Y = 128


def predict_files():
    pm = PredictModel("../save/model/star")
    for i in file_current("/WORK/datasset/flower_photos_test/roses/"):
        result = pm.predict(i, IMG_X, IMG_Y)
        for re in result:
            print(tf.argmax(re))


def evaluate():
    pm = PredictModel("../save/model/flower")
    creator = DatasetCreator(image_x=IMG_X, image_y=IMG_Y)
    ds = creator.load(
        "/WORK/datasset/flower_photos_test/").batch(batch=10).get()
    result = pm.evaluate(ds)
    print(result)


if __name__ == '__main__':
    predict_files()
    # evaluate()
