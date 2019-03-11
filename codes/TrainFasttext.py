#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 00:05:49 2018

@author: jimmy
"""

import fasttext
from timer import timer
import pandas as pd
import numpy as np
import time,os



def get_list(item):
    l = []
    l.append(item.prefix)
    if len(item.query_prediction)>0:
        for d in item.query_prediction:
            l.append(d)
    l.append(item.title)
    return l


def prefix_title_similarity(l,model):
    return model.cosine_similarity(l[0],l[-1])
    

def prefix_query_prediction_similarity(l,model):
    if len(l)==2:
        return []
    result_list = []
    for each in l[1:-1]:
        result_list.append(model.cosine_similarity(l[0],each))
    return result_list
  


def query_prediction_title_similarity(l,model):
    if len(l)==2:
        return []
    result_list = []
    for each in l[1:-1]:
        result_list.append(model.cosine_similarity(l[-1],each))
    return result_list


def write_cps(train_data_,test_data_):
    train_data = train_data_.copy()
    train_data['l'] = train_data.apply(get_list,axis=1)
    with open('corpus_new.txt','a') as f:
        for a in train_data.l:
            f.write(' '.join(a)+' __label__1\n')


def TrainFasttext(data_):
    data = data_.copy()

    print("-"*50)
    print('Begining Train Fasttext......')
    t1 = time.time()
    
    
    with timer("构建词列表特征"):
        data['l'] = data.apply(get_list,axis=1)
    
    
    with timer("构建语料txt"):
        with open('corpus_new.txt','a') as f:
            for a in data.l:
                f.write(' '.join(a)+' __label__1\n')
    
    
    with timer("下载或读取model.bin"):
        if os.path.exists('model.bin'):
            classifier = fasttext.load_model('model.bin')
        else:
            classifier = fasttext.skipgram("corpus_new.txt", "model",dim = 50)
    
    with timer("计算prefix和title的相似度"):
        data['prefix_title_similarity'] = data.l.apply(prefix_title_similarity,model=classifier)

    with timer("计算prefix和query_prediction的相似度"):
        data['prefix_query_prediction_similarity'] = data.l.apply(prefix_query_prediction_similarity,model=classifier)

        data['max_p_q'] = data.prefix_query_prediction_similarity.apply(lambda x:np.max(x) if len(x)>0 else 0)
        
        data['min_p_q'] = data.prefix_query_prediction_similarity.apply(lambda x:np.min(x) if len(x)>0 else 0)
        
        data['mean_p_q'] = data.prefix_query_prediction_similarity.apply(lambda x:np.mean(x) if len(x)>0 else 0)

        data.drop(['prefix_query_prediction_similarity'],axis=1,inplace=True)
        
    
    
    with timer("计算query_prediction和title的相似度"):
        data['query_prediction_title_similarity'] = data.l.apply(query_prediction_title_similarity,model=classifier)

        data['max_q_t'] = data.query_prediction_title_similarity.apply(lambda x:np.max(x) if len(x)>0 else 0)
        
        data['min_q_t'] = data.query_prediction_title_similarity.apply(lambda x:np.min(x) if len(x)>0 else 0)
        
        data['mean_q_t'] = data.query_prediction_title_similarity.apply(lambda x:np.mean(x) if len(x)>0 else 0)
    
        data.drop(['query_prediction_title_similarity','l'],axis=1,inplace=True)
    
    

    print('-'*50)
    print("{} - done in {:.0f}s".format('Train Fasttext', time.time() - t1))

    return data
