# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
if __name__ == '__main__':
    x=["中厚层白云岩,R001003425_中厚层白云岩.jpg","中厚层白云岩,R001003437_中厚层白云岩.jpg"]
    print(x)
    re=map(lambda x:x.split(","),x)
    print(list(re))
