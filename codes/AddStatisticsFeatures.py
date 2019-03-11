#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 00:04:25 2018

@author: jimmy
"""

import pandas as pd
import numpy as np
import itertools
from timer import timer
from reduce_mem_usage import reduce_mem_usage
import time 



### 统计'prefix', 'title', 'tag'组合的统计量
def cal(data,items=['prefix','query_prediction_num'],merge=False,t=1):  
    group_col = []
    for i in range(t,len(items)+1):
        for col in itertools.combinations(items, i):
            group_col.append(list(col))
    
    
    base_data = data[data.flag.isin([1,2])] if merge else data[data.flag==1]
    
    
    for item in group_col:
        temp = base_data.groupby(item, as_index=False)['label']\
                .agg({'_'.join(item)+'_click': 'sum',\
                      '_'.join(item)+'_count':'count'})
        temp['_'.join(item)+'_ctr'] = (temp['_'.join(item)+'_click']/temp['_'.join(item)+'_count']).fillna(0)
        temp['_'.join(item)+'_var'] = (temp['_'.join(item)+'_click']*(temp['_'.join(item)+'_count']-temp['_'.join(item)+'_click'])/(temp['_'.join(item)+'_count']**3)).fillna(0)*1000000
        
        data = pd.merge(data, temp, on=item, how='left')
    
    return data
    



### query_prediction 最大概率
def get_max_pro(d):
    if len(d):
        return d[max(d,key=d.get)]
    else:
        return 0


### 对prefix连续计数
def count_num(data):
    df = data.copy()
    nrow = len(df)
    count = list(range(nrow))
    count[0] = 1
    for i in range(1,nrow):
        if df.prefix[i]==df.prefix[i-1]:
            count[i] = count[i-1]+1
        else:
            count[i] = 1
    df['count'] = count
    
    count1 = np.array(count)
    for i in range(len(count1)):
        if count1[i]>1:
            count1[(i-count1[i]+1):(i+1)] = count1[i]
    df['count1'] = count1
    
    return df




def cal_order(data):
    data['order'] = (np.array(range(len(data)))+1)/len(data)
    return data





def low(a,b):
    c = {}
    if len(a)==0:
        return 0
    
    for i,j in a.items():
        c[i.lower()] = j

    if b.lower() in c:
        if float(c[b.lower()])>0.1:
            return 1
        else:
            return 0
    else:
        return 0







def AddStatisticsFeatures(data_,merge=False):
    
    print("-"*50)
    print('Begining Statistics Feature Add......')
    t1 = time.time()
    
    data = data_.copy()

       
    ### query_prediction 最大概率
    with timer("query_prediction 最大概率"):
        data['max_p'] = data.query_prediction.map(get_max_pro)
    
    ### prefix/title
    with timer("prefix/title比率"):
        data['pt_per'] = data.prefix.apply(len)/data.title.apply(len)

    
    ### prefix 长度
    with timer("prefix 长度"):
        data['prefix_num'] = data.prefix.map(len)

        
    ### query_prediction 个数
    with timer("query_prediction 个数"):
        data['query_prediction_num'] = data.query_prediction.map(len)

    
    ### title 长度
    with timer("title 长度"):
        data['title_num'] = data.title.map(len)

    
    ### tag 长度
    with timer("tag 长度"):
        data['tag_num'] = data.tag.map(len)

    
    
    ### 对prefix连续计数
    with timer("对prefix连续计数"):
        data = count_num(data)

    

    
    ### 统计'prefix','tile','tag'组合的统计量
    with timer("统计'prefix','tile','tag'组合的统计量"):
        data = cal(data,items=['prefix','title','tag'],merge=merge,t=1)

    
    
    
    ### 统计'prefix_num','tag_num','title_num'组合的统计量
    with timer("统计'prefix_num','tag_num','title_num'组合的统计量"):
        data = cal(data,items=['prefix_num','tag_num','title_num'],merge=merge,t=1)
    
    
    
    ### 统计'num'和标签组合的统计量
    with timer("统计'num'和标签组合的统计量"):
        data = cal(data,items=['prefix_num','title_num','tag'],merge=merge,t=3)
        data = cal(data,items=['prefix_num','tag_num','title'],merge=merge,t=3)
        data = cal(data,items=['tag_num','title_num','prefix'],merge=merge,t=3)
        data = cal(data,items=['prefix_num','tag'],merge=merge,t=2)
        data = cal(data,items=['query_prediction_num','tag'],merge=merge,t=2)



    ### 统计'prefix','title','tag','query_prediction_num'的统计量
    with timer("统计'prefix','title','tag','query_prediction_num'的统计量"):
        data = cal(data,items=['prefix','title','tag','query_prediction_num'],merge=merge,t=4)

    
    ### 计算query_prediction的最低统计量
    with timer("计算query_prediction的最低统计量"):
        data['in_query_big'] = data.apply(lambda x:low(x['query_prediction'],x['title']),axis=1)

    
    ### 对tag进行类别转换
    with timer("对tag进行类别转换"):
        data.tag = pd.factorize(data.tag)[0]

    
    ### 压缩数据内存
    with timer("压缩数据内存"):
        data = reduce_mem_usage(data)

    
    
    print('-'*50)
    print("{} - done in {:.0f}s".format('Add Statistics Feature', time.time() - t1))
    
    return data


