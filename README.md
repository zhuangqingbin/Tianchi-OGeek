# 天池大数据竞赛平台-OGeek算法挑战赛（实时搜索场景下搜索结果ctr预估）
[大赛链接](https://tianchi.aliyun.com/competition/entrance/231688/introduction?spm=5176.12281915.0.0.72c510bdrS25xw)
## 数据集
训练数据：200万；验证数据：5万；测试数据1:5万；测试数据2:25万；数据位于`data`文件夹

变量 | 备注说明   
-|-
prefix | 用户输入query前缀 | 
query_prediction | 根据当前前缀，预测的用户完整需求查询词，最多10条 | 
title | 文章标题 | 
tag | 文章内容标签 |
label | 是否点击 |

## 特征工程

- 将整个文本数据（前缀+词频特征的降序排列）结巴分词，利用fasttext训练出前缀和词频特征的最大相似度、最小相似度平均相似度。
代码位于`codes/TrainFasttext.py`

- 统计词频的最大概率、前缀/标题的比率、前缀长度、词频个数、对前缀连续计数、统计前缀标题标签组合的统计量。代码位于`codes/AddStatisticsFeatures.py`

- 构建前缀和文章标题的编辑距离特征（一个字串转化成另一个字串最少的操作次数，在其中的操作包括插入、删除、替换），
前缀和词频的编辑距离特征；计算标题对前缀中概率最大句子的词、字级别的召回率、精确率。代码位于`codes/AddTextFeatures.py`


## 模型训练

- 尝试过FFM模型，效果不是很好，原因应该是给的数据特征维度不够

- 最后考虑到数据实在太大，本地计算机性能有限，没有使用xgboost而是使用ligthgbm。代码位于`codes/LgbModel.py`


