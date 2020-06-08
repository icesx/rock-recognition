# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

from model.train_model import FlowerModel

if __name__ == '__main__':
    FlowerModel(train_image_root="/WORK/datasset/flower_photos_train",
                test_image_root="/WORK/datasset/flower_photos_test",
                image_x=128, image_y=128) \
        .load(batch=10) \
        .train(steps_per_epoch=280,
               epochs=100) \
        .save("../save/model/flower")
