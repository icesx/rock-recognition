import tensorflow as tf
from dataset.rock_dataset import RockDataset
from tensorflow import keras


class Rock:
    def __init__(self):
        self.model = None
        self.ds = None

    def load(self):
        self.ds = RockDataset().load('/WORK/datasset/rock_imgs_train2')
        return self

    def train(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(64, (2, 2), activation="relu", input_shape=(128, 128, 3)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (2, 2), activation="relu"),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (2, 2), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(184)
        ])
        self.model.compile(optimizer="adam",
                           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=["accuracy"])
        self.model.summary()
        self.model.fit(self.ds, epochs=100, steps_per_epoch=150)
        return self

    def save(self):
        tf.saved_model.save(self.model, export_dir="../tmp/save/rock")


if __name__ == '__main__':
    from utils.gpu import gpu_init
    gpu_init(6000)
    Rock().load().train().save()
