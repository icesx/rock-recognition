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
1. 在训练的时候出现了损失函数无法下降的问题，导致准确率只有60%左右。，估计是正则化太多了。后去掉多余的正则化，能够正常的收敛
2. 此时出现过拟合的问题：accuracy很快收敛，但是val_accuracy不收敛，val_loss有一个降低又升高的过程。val_accuracy最高能够达到0.7的样子
3. 通过数据增强后，val_accuracy增加到0.8左右，但是仍然会过拟合。
4. 增加dropout和l2后，效果并不明显
5. 修改batch from 28 到10后，波动变小，收敛变慢
6. 再次减小l2后，收敛变快，又再次过拟合
### predict
```python
python3 predict/rock_predict_tensor.py
python3 predict/star_predict_tensor.py
```
