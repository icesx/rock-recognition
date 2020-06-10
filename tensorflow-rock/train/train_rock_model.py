from dataset.dataset_group import CustomDatasetGroup
from model.model_creater import BaseModelOperate
import tensorflow as tf
from tensorflow import keras


def __define_model(image_x, image_y):
    model = keras.Sequential([
        keras.layers.Conv2D(120, (2, 2), activation="relu", input_shape=(image_x, image_y, 3)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (2, 2), activation="relu"),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(32, (2, 2), activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(167)
    ])
    return model


if __name__ == '__main__':
    BaseModelOperate(CustomDatasetGroup(train_root="/WORK/datasset/rock_imgs_train",
                                        validation_root="/WORK/datasset/rock_imgs_test",
                                        image_x=128,
                                        image_y=128),
                     name="Rock").train(batch=10,
                                        steps_per_epoch=150,
                                        epochs=300,
                                        validation_steps=20,
                                        evaluate_steps=20,
                                        fun_provide_model=__define_model)
