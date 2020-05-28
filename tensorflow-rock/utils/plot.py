# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import matplotlib.pyplot as plt


def plot_show(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.show()


def plot_shows(train_images, train_labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])
    plt.show()
