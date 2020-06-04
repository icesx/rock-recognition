# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import pathlib
import shutil
import os


def write(file_path, file_content):
    with open(file_path, 'w') as file:
        file.write(file_content)


def over_write(file_path, file_content):
    with open(file_path, "w") as file:
        file.seek(0)
        file.write(file_content)


def copy(file_path, target_path):
    print("copy ", file_path, " to ", target_path)
    shutil.copy2(file_path, target_path)


def mk_dir(dir):
    print("mk_dir ", dir)
    if os.path.exists(dir) is False:
        os.mkdir(dir)


def file_list(root_path):
    s = '*/*'
    return __list_file(root_path, s)


def file_current(root_path):
    s = './*'
    return __list_file(root_path, s)


def __list_file(root_path, s):
    root_path = pathlib.Path(root_path)
    all_image_paths = list(root_path.glob(s))
    return [str(path) for path in all_image_paths]
