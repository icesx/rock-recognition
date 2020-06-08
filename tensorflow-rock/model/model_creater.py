# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from dataset.create_dataset import DatasetCreator
from utils.tf_board import tf_board
import tensorflow as tf
from tensorflow import keras
from dataset.image_file import ALL_LABELS


class BaseModelOperate:
    def __init__(self, image_x, image_y):
        from utils.tf_gpu import gpu_init
        gpu_init(6000)
        self.__model = None
        self.__ds_train = None
        self.__ds_test = None
        self.__label_file = "../save/lables_{0}.csv".format(self.__class__.__name__)
        self.__image_x = image_x
        self.__image_y = image_y

    def load(self, train_image_root, test_image_root, batch=10):
        self.__ds_train = self.__dataset(batch, train_image_root)
        self.__ds_test = self.__dataset(batch, test_image_root)
        self.__write_labels()
        return self

    def __dataset(self, batch, image_root):
        creator = DatasetCreator(image_x=self.__image_x, image_y=self.__image_y)
        return creator.load(
            image_root).batch(batch).get()

    def __write_labels(self):
        with open(self.__label_file, 'w') as file:
            for label in ALL_LABELS.items():
                file.write(label[0] + "," + str(label[1].label_idx) + "\r\n")

    def train(self, steps_per_epoch, epochs=100, validation_steps=28):
        self.__model = self._provide_model(self.__image_x, self.__image_y)
        if self.__model is None:
            print("please overrider _train()")
        else:
            self.__compile(self.__model)
            self.__model.summary()
            self._fit(self.__ds_train, self.__model, epochs, steps_per_epoch, validation_steps)
            self.__evaluete(self.__model, self.__ds_test.repeat())
            return self

    def _provide_model(self, image_x, image_y):
        pass

    def save(self, save_dir):
        tf.saved_model.save(self.__model, export_dir=save_dir)

    def _fit(self, dataset, model, epochs, steps_per_epoch, validation_steps):
        model.fit(dataset,
                  epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=self.__ds_test,
                  validation_steps=validation_steps,
                  callbacks=[tf_board(self.__class__.__name__)])

    def __evaluete(self, model, test_dataset):
        test_loss = model.evaluate(test_dataset, steps=10)
        print(test_loss)

    def __compile(self, model):
        model.compile(optimizer="adam",
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"
                               # , tf.keras.metrics.Recall(class_id=0)
                               ])
