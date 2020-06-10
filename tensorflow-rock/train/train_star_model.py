from dataset.dataset_group import CustomDatasetGroup
from model.model_creater import BaseModelOperate

from tensorflow import keras
import tensorflow as tf


def _provide_model(image_y, image_x):
    model = keras.Sequential([
        keras.layers.Conv2D(10, (5, 5), activation=tf.nn.relu, input_shape=(image_y, image_x, 3),
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.001)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(20, (3, 3),
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.001)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.001)),
        keras.layers.Dense(10)
    ])
    return model


if __name__ == '__main__':
    BaseModelOperate(CustomDatasetGroup(train_root="/WORK/datasset/star_imgs_train",
                                        validation_root="/WORK/datasset/star_imgs_test",
                                        image_x=128, image_y=128), name="Start") \
        .train(batch=21,
               steps_per_epoch=300,
               epochs=20,
               validation_steps=21,
               fun_provide_model=_provide_model)
