# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import threading


class IdGenerator(object):
    def __init__(self):
        self.cur_id = 0
        self.lock = threading.Lock()

    def next_id(self):
        with self.lock:
            result = self.cur_id
            self.cur_id += 1
        return result
