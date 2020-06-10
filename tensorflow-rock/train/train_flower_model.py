# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

from tensorflow import keras
import tensorflow as tf

from dataset.dataset_group import CustomDatasetGroup
from model.model_creater import BaseModelOperate


def _provide_model(image_y, image_x):
    model = keras.Sequential([
        keras.layers.Conv2D(18, (3, 3),
                            activation=tf.nn.relu,
                            input_shape=(image_y, image_x, 3)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3),
                            activation=tf.nn.relu,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.002)),
        keras.layers.Conv2D(64, (3, 3),
                            activation=tf.nn.relu,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.002)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3),
                            activation=tf.nn.relu,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.002)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3),
                            activation=tf.nn.relu,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.002)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64,
                           activation=tf.nn.relu,
                           kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                           bias_regularizer=tf.keras.regularizers.l2(l=0.002)),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(5)
    ])
    return model


if __name__ == '__main__':
    BaseModelOperate(CustomDatasetGroup(train_root="/WORK/datasset/flower_photos_train",
                                        validation_root="/WORK/datasset/flower_photos_test",
                                        image_x=224, image_y=224),
                     name="FlowerModel").train(batch=28,
                                               steps_per_epoch=100,
                                               epochs=2,
                                               validation_steps=30,
                                               fun_provide_model=_provide_model,
                                               )
