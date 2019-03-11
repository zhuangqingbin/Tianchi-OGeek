#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:52:21 2018

@author: jimmy
"""

import pandas as pd
import numpy as np
import json
from timer import timer
import time,re

def str_to_dict(x):
    try:
        dic = json.loads(x)
        for d in dic:
            dic[d] = float(dic[d])
        return dic
    except:
        return {}
    







def read_data(train_path,valid_path,test_path):
    print("-"*50)
    print('Begining read data......')
    t1 = time.time()
   
    col_names = ['prefix','query_prediction','title','tag','label']
    
    train = pd.read_table(train_path,names = col_names,header=None,encoding='utf8',quoting=3).astype(str)
    train.label = train.label.fillna(-1).astype(np.int8)
    
    valid = pd.read_table(valid_path,names = col_names,header=None,encoding='utf8',quoting=3).astype(str)
    
    valid.label = valid.label.fillna(-1).astype(np.int8)
    
    test = pd.read_table(test_path,names = col_names,header=None,encoding='utf8',quoting=3).astype(str)
    test.label = test.label.map({'nan':1}).astype(np.int8)
    
    
    train['prefix'] = train['prefix'].map(lambda x : str(x).lower().strip())
    valid['prefix'] = valid['prefix'].map(lambda x : str(x).lower().strip())
    test['prefix'] = test['prefix'].map(lambda x : str(x).lower().strip())
    
    train['title'] = train['title'].map(lambda x : str(x).lower().strip())
    valid['title'] = valid['title'].map(lambda x : str(x).lower().strip())
    test['title'] = test['title'].map(lambda x : str(x).lower().strip())
    
    train.query_prediction = train.query_prediction.map(str_to_dict)
    valid.query_prediction = valid.query_prediction.map(str_to_dict)
    test.query_prediction = test.query_prediction.map(str_to_dict)
    
    train['flag'] = 1
    valid['flag'] = 2
    test['flag'] = 3
    
    print('-'*50)
    print("{} - done in {:.0f}s".format('Read data', time.time() - t1))
    
    return pd.concat([train,valid,test]).reset_index(drop=True)
    
