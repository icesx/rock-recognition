from dataset.dataset_group import CustomDatasetGroup
from model.model_creater import BaseModelOperate

from tensorflow import keras
import tensorflow as tf


def mnist(image_y, image_x):
    return keras.Sequential([
        keras.layers.Conv2D(10, (5, 5), activation=tf.nn.relu, input_shape=(image_y, image_x, 3),
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.001)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.001)),
        keras.layers.Dense(10)
    ])


if __name__ == '__main__':
    BaseModelOperate(CustomDatasetGroup(train_root="/WORK/datasset/mnist_train",
                                        validation_root="/WORK/datasset/mnist_validation",
                                        image_x=28,
                                        image_y=28),
                     name="Mnist").train(batch=20,
                                         steps_per_epoch=3000,
                                         epochs=5,
                                         validation_steps=21,
                                         fun_provide_model=mnist)
