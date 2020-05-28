from pathlib import Path
from typing import List

import pathlib
from pathlib import PosixPath
import pandas as pd
import shutil


def group_image(root_dir):
    data_root = pathlib.Path(root_dir)
    all_image_paths: List[PosixPath] = list(data_root.glob('*/*'))
    lables = list(map(lambda f: str(f.parent), all_image_paths))
    df = pd.DataFrame({"lables": lables, "count": [1 for i in range(len(lables))]})
    print(df)
    dfg = df.groupby("lables")
    df_sum = dict(dfg["count"].sum())
    print(df_sum)
    df_result = {(k,v) for k, v in df_sum.items() if v > 20}
    for (k,v) in df_result:
        print(k,v)
        shutil.copytree(k,dst="/WORK/datasset/rock_imgs_train/"+k.split("/")[4])



if __name__ == '__main__':
    group_image("/WORK/datasset/rock_imgs")
