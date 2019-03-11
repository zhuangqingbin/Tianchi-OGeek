#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:57:40 2018

@author: jimmy
"""
import os,sys,gc
sys.path.append('codes')
sys.path.append('data')


from DataRead import read_data
import pandas as pd
import numpy as np
import itertools,time

from AddTextFeatures import AddTextFeatures
from AddStatisticsFeatures import AddStatisticsFeatures
from TrainFasttext import TrainFasttext
from LgbModel import train1
from configure import threshold_list_in,threshold_list_out,params_in,params_out


if os.path.exists('data/train_data.pkl') & os.path.exists('data/valid_data.pkl') & os.path.exists('data/test_data.pkl'):
    train_data_ = pd.read_pickle('data/train_data.pkl')
    valid_data_ = pd.read_pickle('data/valid_data.pkl')
    test_data_ = pd.read_pickle('data/test_data.pkl')
    
else:
    ###读取数据
    
    train_path="data/oppo_round1_train_20180929.txt"
    valid_path="data/oppo_round1_vali_20180929.txt"
    test_path="data/oppo_round1_test_B_20181106.txt"
    data = read_data(train_path=train_path,valid_path=valid_path,test_path=test_path)

    ###数据处理
    data1 = AddTextFeatures(data)
    data2 = AddStatisticsFeatures(data1,merge=False)
    data3 = TrainFasttext(data2)


    train_data_ = data3[data3.flag==1].drop(['prefix','query_prediction','title','flag'],axis = 1)
    valid_data_ = data3[data3.flag==2].drop(['prefix','query_prediction','title','flag'],axis = 1)       
    test_data_ = data3[data3.flag==3].drop(['prefix','query_prediction','title','flag'],axis = 1)

    del data1,data2,data3
    gc.collect()
    train_data_.to_pickle('data/train_data.pkl')
    valid_data_.to_pickle('data/valid_data.pkl')
    test_data_.to_pickle('data/test_data.pkl')



### 开始训练
#train(params_in = params_in,params_out = params_out,train_data_=train_data_,valid_data_=valid_data_,test_data_=test_data_,threshold_list_in=threshold_list_in,threshold_list_out=threshold_list_out,write=False)

train1(params_in = params_in,params_out = params_out,train_data = pd.concat([train_data_,valid_data_]).reset_index(drop=True),test_data=test_data_.reset_index(drop=True),t=2,N = 10,threshold_list_in=threshold_list_in,threshold_list_out=threshold_list_out,random_state=42,write=False)



