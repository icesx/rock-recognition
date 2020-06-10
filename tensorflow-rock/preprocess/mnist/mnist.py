# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from tensorflow import keras
import tensorflow as tf
from utils.plot import PlotContext
import matplotlib as plot
from utils.my_file import mk_dir


def train_dataset(root, numpy_image, numpy_label):
    pc = PlotContext()
    i = 0
    for image, label in zip(numpy_image, numpy_label):
        i += 1
        if i >= 25:
            pc.show()
        else:
            print("Image", image)
            print("Label:", label)
            pc.append(image, label)
        folder = root + "/" + str(label)
        mk_dir(folder)
        plot.image.imsave(folder + "/" + str(i) + ".jpg", image)


if __name__ == '__main__':
    train_root = "/WORK/datasset/mnist_train/"
    validation = "/WORK/datasset/mnist_validation/"
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_dataset(train_root, train_images, train_labels)
    train_dataset(validation, test_images, test_labels)
