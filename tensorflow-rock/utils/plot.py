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


class PlotContext:
    def __init__(self, title="plot_title", nrows=5, ncols=5):
        plt.figure(figsize=(10, 10))
        plt.title(title)
        self.__index = 0
        self.__nrows = nrows
        self.__ncols = ncols

    def append(self, image, label):
        if self.__index + 1 > self.__ncols * self.__nrows:
            print("your set nrow={0},ncols={1},but set {2} image".format(self.__nrows, self.__ncols, self.__index))
        else:
            plt.subplot(self.__nrows, self.__ncols, self.__index + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image, cmap=plt.cm.binary)
            plt.xlabel(label)
            self.__index += 1

    def show(self):
        plt.show()
