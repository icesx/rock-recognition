# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

from model.train_model import FlowerModel, Flower102Model

if __name__ == '__main__':
    Flower102Model(image_x=128, image_y=128) \
        .load(train_image_root="/WORK/datasset/102flowers_train",
              test_image_root="/WORK/datasset/102flowers_test", batch=20) \
        .train(steps_per_epoch=300,
               epochs=100) \
        .save("../save/model/102flower")
