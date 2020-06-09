# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from tensorflow.python.keras.callbacks import LearningRateScheduler

from dataset.create_dataset import CustomDatasetCreator
from dataset.dataset_group import DatasetGroup
from utils.tf_board import tf_board
import tensorflow as tf
from tensorflow import keras
from dataset.image_file import ALL_LABELS
import tensorflow_datasets as tfds


def _fit(dataset,
         validation_ds,
         model,
         epochs,
         steps_per_epoch,
         validation_steps,
         tf_board_name):
    return model.fit(dataset,
                     epochs=epochs,
                     steps_per_epoch=steps_per_epoch,
                     validation_data=validation_ds,
                     validation_steps=validation_steps,
                     callbacks=[tf_board(tf_board_name)])


def _compile(model):
    optimizer_1 = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer_1,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"
                           # , tf.keras.metrics.Recall(class_id=0)
                           ])


def _save(model, save_dir):
    tf.saved_model.save(model, export_dir=save_dir)


def _evaluete(model, test_dataset, evalue_step=10):
    test_loss = model.evaluate(test_dataset, evalue_step)
    print(test_loss)


def _write_labels(label_file):
    with open(label_file, 'w') as file:
        for label in ALL_LABELS.items():
            file.write(label[0] + "," + str(label[1].label_idx) + "\r\n")


class BaseModelOperate(object):
    def __init__(self, dataset_group, name):
        self.dataset_group = dataset_group
        self.__name = name
        _write_labels("../save/" + name + ".csv")

    def train(self, batch,
              steps_per_epoch,
              epochs=100,
              validation_steps=28,
              evaluete_steps=10,
              fun_provide_model=None,
              fun_compile=_compile,
              fun_fit=_fit,
              fun_save_model=_save,
              evaluete=_evaluete
              ):
        ds_train = self.dataset_group.train(batch)
        ds_test = self.dataset_group.validation(batch)
        __model = fun_provide_model(self.dataset_group.image_y, self.dataset_group.image_x)
        if __model is None:
            print("please overrider _train()")
        else:
            fun_compile(__model)
            __model.summary()
            history = fun_fit(ds_train, ds_test, __model, epochs, steps_per_epoch, validation_steps, self.__name)
            fun_save_model(__model, "../save/model/" + self.__name)
            evaluete(__model, ds_test, evaluete_steps)
            return self
