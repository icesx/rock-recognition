# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("../rock_uniq.txt", sep=",")
    print(df)
    df_g = df.groupby(by="name")
    for id, name in df_g:
        print(name,id)
