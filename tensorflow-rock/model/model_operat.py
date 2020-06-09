# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com


class AbsModelOperate:
    def __init__(self,
                 model_create,
                 model_compile,
                 model_fit,
                 model_save,
                 save_label,
                 ):
        from utils.tf_gpu import gpu_init
        gpu_init(6000)
        self.model_create = model_create
        self.model_compile = model_compile
        self.model_fit = model_fit
        self.model_save = model_save
        self.save_label = save_label

    def run(self):
        model = self.model_create()
        model = self.model_compile(model)
        self.save_label(self.__label_file)
        model = self.model_fit(model)
        self.model_save(model)
