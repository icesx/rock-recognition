tensorflow卷积网络训练的石头识别的模型
======
### dataset
>使用spinder_rock爬取。

>人工进行素材的筛选，初略的删除了一些明显不对的石头，或者像素很低的

>使用如下方法来切割数据集

```python
python3 test/test_split_dataset.py
```
### tensorboard
```
python3 -m tensorboard.main --logdir=./tmp/logs/tb/
```
### 模型训练

```python
python3 train/train_rock_model.py
```
### predict
```python
python3 predict/rock_predict_tensor.py
python3 predict/star_predict_tensor.py
```
