# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from dataset.dataset_group import CustomDatasetGroup
from model.model_creater import BaseModelOperate
from tensorflow import keras
import tensorflow as tf


def _model(image_y, image_x):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3),
                            activation=tf.nn.relu, input_shape=(image_y, image_x, 3)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3),
                            activation=tf.nn.relu, ),
        keras.layers.MaxPool2D((2, 2)),
        # keras.layers.Conv2D(128, (3, 3),
        #                     activation=tf.nn.relu, ),
        # keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        # keras.layers.Dense(512,
        #                    activation=tf.nn.relu,
        #                    kernel_regularizer=tf.keras.regularizers.l2(l=0.005),
        #                    bias_regularizer=tf.keras.regularizers.l2(l=0.005)),
        # keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(256,
                           activation=tf.nn.relu,
                           kernel_regularizer=tf.keras.regularizers.l2(l=0.005),
                           bias_regularizer=tf.keras.regularizers.l2(l=0.005)),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(102)
    ])
    return model


if __name__ == '__main__':
    image_x = 128
    image_y = 128
    BaseModelOperate(CustomDatasetGroup(train_root="/WORK/datasset/102flowers_train",
                                        validation_root="/WORK/datasset/102flowers_test",
                                        image_y=image_y,
                                        image_x=image_x),
                     name="Flower102Model",
                     ).train(batch=30,
                             steps_per_epoch=200,
                             epochs=50,
                             validation_steps=30,
                             evaluate_steps=10,
                             fun_provide_model=_model)
