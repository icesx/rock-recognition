# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
<<<<<<< HEAD
<<<<<<< HEAD
import pathlib
import shutil


def write(file_path, file_content):
=======


def file_save(file_path, file_content):
>>>>>>> c35f7f48f7b3e88259866ff6f548f1ab53f7cbe3
=======
import pathlib
import shutil
import os


def write(file_path, file_content):
>>>>>>> 410dda6... remove model step-1
    with open(file_path, 'w') as file:
        file.write(file_content)


<<<<<<< HEAD
<<<<<<< HEAD
def over_write(file_path, file_content):
    with open(file_path, "w") as file:
        file.seek(0)
        file.write(file_content)


def copy(file_path, target_path):
    print("copy ", file_path, " to ", target_path)
    shutil.copy2(file_path, target_path)


def file_list(root_path):
    root_path = pathlib.Path(root_path)
    all_image_paths = list(root_path.glob('./*'))
    return [str(path) for path in all_image_paths]
=======
def file_resave(file_path, file_content):
    with open(file_path, "w") as file:
        file.seek(0)
        file.write(file_content)
>>>>>>> c35f7f48f7b3e88259866ff6f548f1ab53f7cbe3
=======
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
    root_path = pathlib.Path(root_path)
    all_image_paths = list(root_path.glob('./*'))
    return [str(path) for path in all_image_paths]
>>>>>>> 410dda6... remove model step-1
