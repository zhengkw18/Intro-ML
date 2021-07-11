# Code

`simple/`: 存储MLP、CNN、RNN、双向LSTM的训练代码

- 下载Glove 6B预训练词向量，将`glove.6B.50d.txt`放在目录下；`train.csv`也放在目录下
- 运行`transform.py`，进行数据集处理和嵌入，保存为`npy`文件
- 在`train.py`中更改`net_type`，运行即可训练、验证

`classification/`,`regression/`: 分别存放BERT的分类、回归代码

- 由于模型较大，笔者使用Google Colab进行训练，GPU为P100/V100，显存为16G。`drive_dir`表示挂载目录，可调用以下命令挂载

  ```python
  from google.colab import drive
  drive.mount('/content/drive/')
  ```

  并使用`pip install transformers`安装包

- `pretrained`变量代表预训练模型，`SEQ_LEN`为截断长度。

- 运行`bert.py`进行分词和训练，程序会自动保存checkpoint

- 运行`bert_infer.py`得到预测结果，保存在`predict.csv`中

- 若显存不足，需要调小模型大小、截断长度或batch size

`regression/postprocess.py`是一个简单的集成程序，读取若干预测结果并取平均。

# Contact

郑凯文 2018011314 1048198090@qq.com

**Kaggle Id**: `Kaiwen Zheng123`

