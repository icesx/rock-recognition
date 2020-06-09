# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

from tensorflow import keras
import tensorflow as tf

from dataset.dataset_group import CustomDatasetGroup
from model.model_creater import BaseModelOperate


def _provide_model(image_x, image_y):
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3, 3),
                            activation=tf.nn.relu,
                            input_shape=(image_x, image_y, 3)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3),
                            activation=tf.nn.relu),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(512,
                           activation=tf.nn.relu,
                           kernel_regularizer=tf.keras.regularizers.l2(l=0.05),
                           bias_regularizer=tf.keras.regularizers.l2(l=0.01)),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(128,
                           activation=tf.nn.relu,
                           kernel_regularizer=tf.keras.regularizers.l2(l=0.03),
                           bias_regularizer=tf.keras.regularizers.l2(l=0.05)),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(5)
    ])
    return model


if __name__ == '__main__':
    BaseModelOperate(CustomDatasetGroup(train_root="/WORK/datasset/flower_photos_train",
                                        validation_root="/WORK/datasset/flower_photos_test",
                                        image_x=128, image_y=128),
                     name="FlowerModel").train(batch=30,
                                               steps_per_epoch=280,
                                               epochs=100,
                                               validation_steps=30,
                                               provide_model_fun=_provide_model,
                                               )
