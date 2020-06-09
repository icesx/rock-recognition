# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from scipy.io import loadmat
import csv
from utils.my_file import copy, mk_dir


def mat_csv():
    mat = loadmat("imagelabels.mat")
    j = 1
    with open("flower_name.csv", "r") as flower_name:
        name_dic = dict((line[0], line[1]) for line in csv.reader(flower_name))

    with open("label.csv", "w") as lable_csv:
        for i in mat["labels"][0]:
            name_id=str(i)
            lable_csv.write(str(j) + "," + name_id + "," + name_dic.get(name_id)+"\r\n")
            j += 1


def split_flower():
    target = "/WORK/datasset/102flowers/"
    root = "/WORK/datasset/102flowers_org/"
    with open("label.csv") as csv_file:
        data = csv.reader(csv_file)
        for row in data:
            image = "image_{0}.jpg".format("%05d" % int(row[0]))
            label = str(row[2])
            folder_label = target + "" + label
            mk_dir(folder_label)
            copy(root + image, folder_label + "/" + image)


if __name__ == '__main__':
    # mat_csv()
    split_flower()
