# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import os
import pathlib
import shutil


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
    return __file_list(root_path, './*')


def file_list_recursive(root_path, file_type="*"):
    return __file_list(root_path, './**/' + file_type)


def __file_list(root_path, pattern):
    if os.path.exists(root_path) is False:
        print("WARN: cannot find root path %s" % root_path)
        return []
    root_path = pathlib.Path(root_path)
    all_image_paths = list(root_path.glob(pattern))
    return [str(path) for path in all_image_paths]
