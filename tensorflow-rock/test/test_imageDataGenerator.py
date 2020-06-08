# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow as tf
from utils.tf_board import tf_board
if __name__ == '__main__':
    from utils.tf_gpu import gpu_init

    gpu_init(6000)
    data_gen = ImageDataGenerator(rescale=1. / 255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
    train_it = data_gen.flow_from_directory("/WORK/datasset/102flowers_train",
                                            batch_size=32,
                                            target_size=(128, 128))
    val_it = data_gen.flow_from_directory("/WORK/datasset/102flowers_test",
                                          batch_size=32,
                                          target_size=(128, 128))
    print(train_it.next())
    print(train_it.class_indices)
    print(train_it.labels)
    model = keras.Sequential([
        keras.layers.Conv2D(120, (2, 2), activation="relu", input_shape=(128, 128, 3)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (2, 2), activation="relu"),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(32, (2, 2), activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(102)
    ])
    model.summary()
    model.compile(optimizer="adam",
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"
                           # , tf.keras.metrics.Recall(class_id=0)
                           ])
    model.fit_generator(train_it,
                        steps_per_epoch=100,
                        epochs=50,
                        validation_data=val_it,
                        validation_steps=50,
                        callbacks=[tf_board("TEST_IMAGE")])

