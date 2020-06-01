# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com


def file_save(file_path, file_content):
    with open(file_path, 'w') as file:
        file.write(file_content)


def file_resave(file_path, file_content):
    with open(file_path, "w") as file:
        file.seek(0)
        file.write(file_content)
