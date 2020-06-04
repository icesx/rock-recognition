# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

import tensorflow as tf
from dataset.create_dataset import DatasetCreator
from tensorflow import keras

from utils.my_file import over_write
from utils.tf_board import tf_board
from dataset.image_file import ALL_LABELS


class BaseModelOperate:
    def __init__(self, train_image_root, test_image_root, image_x, image_y):
        from utils.tf_gpu import gpu_init
        gpu_init(6000)
        self.__model = None
        self.__ds_train = None
        self.__ds_test = None
        self.__label_file = "../save/lables_{0}.csv".format(self.__class__.__name__)
        self.__image_x = image_x
        self.__image_y = image_y
        self.__train_image_root = train_image_root
        self.__test_image_root = test_image_root

    def load(self, batch=10):
        self.__ds_train = self.__dataset(batch, self.__train_image_root)
        self.__ds_test = self.__dataset(batch, self.__test_image_root)
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

    def train(self, steps_per_epoch, epochs=100):
        self.__model = self._create(self.__image_x, self.__image_y)
        if self.__model is None:
            print("please overrider _train()")
        else:
            self.compile(self.__model)
            self.__model.summary()
            self.fit(self.__ds_train, self.__model, epochs, steps_per_epoch)
            self.evaluete(self.__model, self.__ds_test.repeat())
            return self

    def save(self, save_dir):
        tf.saved_model.save(self.__model, export_dir=save_dir)

    def fit(self, dataset, model, epochs, steps_per_epoch):
        history = model.fit(dataset,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=self.__ds_test,
                            validation_steps=10,
                            callbacks=[tf_board(self.__class__.__name__)])
        return history

    def evaluete(self, model, test_dataset):
        test_loss = model.evaluate(test_dataset, steps=10)
        print(test_loss)

    def compile(self, model):
        model.compile(optimizer="adam",
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy", tf.keras.metrics.Recall(class_id=0)])


class RockModel(BaseModelOperate):
    def __init__(self, train_image_root, test_image_root, image_x, image_y):
        BaseModelOperate.__init__(self, train_image_root, test_image_root, image_x, image_y)

    def _create(self, image_x, image_y):
        model = keras.Sequential([
            keras.layers.Conv2D(120, (2, 2), activation="relu", input_shape=(image_x, image_y, 3)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (2, 2), activation="relu"),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (2, 2), activation="relu"),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (2, 2), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(167)
        ])
        return model


class StarModel(BaseModelOperate):
    def __init__(self, train_image_root, test_image_root, image_x, image_y):
        BaseModelOperate.__init__(self, train_image_root, test_image_root, image_x, image_y)

    def _create(self, image_x, image_y):
        model = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(image_x, image_y, 3)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(64, activation=tf.nn.sigmoid),
            keras.layers.Dense(10)
        ])
        return model
