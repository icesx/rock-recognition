# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

from tensorflow import keras
from dataset.image_file import *
from dataset.create_dataset import DatasetCreator


def load_model(export_dir):
    print("load model from ", export_dir)
    return keras.models.load_model(export_dir)


def test_dataset(image_path):
    return DatasetCreator(220, 220).load(image_path).batch(1)


def predict(image_path):
    model = load_model("../save/rock")
    ds, image_labels = test_dataset(image_path=image_path)
    # for t, l in enumerate(ds):
    #     print(l[1], image_labels.idx_name_label)

    result = model.predict(ds, steps=5)
    for i, r in enumerate(result):
        print(tf.argmax(r), image_labels.label_name[i],image_labels.idx_name_label[tf.argmax(r).numpy()])


def test_tensor(image_path):
    all_image_bytes, all_image_labels = images_byte_array(image_path, 128, 128)
    print(all_image_labels[0:10])
    predict(all_image_labels[0:10])
    ds = tf.data.Dataset.from_tensor_slices((all_image_bytes, all_image_labels))
    model = load_model("../save/rock")
    print(model)
    result = model.predict(ds, steps=5)
    # for i, r in result:
    #     print(tf.argmax(r), all_image_labels[i])


if __name__ == '__main__':
    from utils.gpu import gpu_init

    gpu_init(6000)
    predict("/WORK/datasset/star_imgs_test")
    # predict("/WORK/datasset/rock_imgs_test")
    # print("test images:", test_images[:10])
    # plot_show(test_images)
    # predicted = model.predict(test_images)
    # for i in predicted:
    #     print(i)
