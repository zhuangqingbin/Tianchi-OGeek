#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:39:10 2018

@author: jimmy
"""
import datetime
from sklearn.model_selection import StratifiedKFold
import itertools
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import f1_score
import time 
from timer import timer



def tr_te(params,train_data,test_data,t=2,N = 10,threshold_list=[0.42],random_state=2018,write=False):

    print('Train beginning.....')
    
    X_tr = np.array(train_data.drop(['label'], axis = 1))
    y_tr = np.array(train_data['label'])
    
    X_te = np.array(test_data.drop(['label'], axis = 1))
    y_te = np.array(test_data['label'])
    
    print('='*50)
    print('The shape of train_data:',X_tr.shape)
    print('The shape of train_label:',y_tr.shape)
    print('The shape of test_data:',X_te.shape)
    print('='*50)
    
    train_logloss = []
    importance = []
    test_submit = []
    dict_f = {}
    for t in threshold_list:
        dict_f[str(t)] = []
    
    skf = StratifiedKFold(n_splits=N, random_state=random_state, shuffle=True)
    
    
    for k, (train_in, test_in) in enumerate(skf.split(X_tr, y_tr)):
        #if k==2:
        #    break
            
        print('-'*50)
        print('Train {} flod.....'.format(k+1))
        X_train, X_test, y_train, y_test = X_tr[train_in], X_tr[test_in], y_tr[train_in], y_tr[test_in]
    
        lgb_train = lgb.Dataset(X_train, y_train,categorical_feature=[0])
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,categorical_feature=[0])
    
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=5000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=50,
                        verbose_eval=50,
                        categorical_feature=[0]
                        )
        pre_p = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        
        
        
        
        
        for t in threshold_list:
            pre = np.where(pre_p > t, 1,0)
            f_score = f1_score(y_test, pre)
            dict_f[str(t)].append(f_score)
            print('Flod {} threshold {}--F1 score:{}\n'.format(k,t,f_score))
        
        logloss = gbm.best_score['valid_0']['binary_logloss']
        
        test_pre_p = gbm.predict(X_te, num_iteration=gbm.best_iteration)
        
        train_logloss.append(logloss)
        importance.append(gbm.feature_importance())
        test_submit.append(test_pre_p)
    
    #### 展示
    
    print('-'*50+'\n')
    
    f_mean_dict = {}
    for t in threshold_list:
        f_mean = np.mean(dict_f[str(t)])
        f_mean_dict[str(t)] = f_mean
        print('Thershold {} f1 score:{}'.format(t,f_mean))
    
    max_t = float(max(f_mean_dict,key=f_mean_dict.get))
    max_f = f_mean_dict[str(max_t)]
    
    print('-'*50+'\n')
    print('Train_data logloss:', np.mean(train_logloss))
    
    
    im = 0
    for i in importance:
        im = im+i
    max_index = np.argsort(im)[::-1]
    im_features = np.array(list(train_data.drop(['label'], axis = 1).columns))[max_index][:5]
    
    te = 0
    for i in test_submit:
        te = te + i

    te_pre = pd.Series(list(te/10)).apply(lambda x: 1 if x >max_t else 0)
    
    if write==False:
        return te_pre.values,round(max_f,4),max_t,im_features
    
    
    ### test_data预测并写出csv
    id = datetime.datetime.strftime(datetime.datetime.now(),'%m-%d-%H-%M')
    file = 'submit%s' % datetime.datetime.strftime(datetime.datetime.now(),'%m%d')
    if os.path.exists(file):
        pass
    else:
        os.mkdir(file)
    

    path_text = '%s/record.txt' % file
    with open(path_text,'a+') as f:
        f.write('ID:{}\t Train F-score:{} \t {} of {} best in train_data, N:{}, p:{}\n '.format(id,max_f,max_t,str(threshold_list),N,train_data.shape[1]))     
        f.write(str(params)+'\n\n')

    path_csv = '{}/{}_Tr{}_t{}.csv'.format(file,id,round(max_f,4),max_t) 
    te_pre.to_csv(path_csv,index = False)
    return te_pre.values,round(max_f,4),max_t,im_features
        


def train1(params_in,params_out,train_data,test_data,t,N,threshold_list_in,threshold_list_out,random_state=42,write=False):
    t1 = time.time()
    
    result = pd.Series(np.zeros(len(test_data)),dtype=np.int8)
    out_index = test_data.prefix_title_tag_ctr.isnull()
    out_num = sum(out_index)
    out_cols = list(test_data[out_index].dropna(axis=1,thresh=int(out_num*0.82)).columns)
    
    result[~out_index],f_in,t_in,im_features_in = tr_te(params=params_in,train_data=train_data,test_data=test_data[~out_index],t=t,N = N,threshold_list = threshold_list_in,random_state=random_state,write=write)
    
    result[out_index],f_out,t_out,im_features_out = tr_te(params=params_out,train_data=train_data[out_cols],test_data=test_data[out_index][out_cols],t=t,N = N,threshold_list = threshold_list_out,random_state=random_state,write=write)
    
    
    ### result预测并写出csv
    id = datetime.datetime.strftime(datetime.datetime.now(),'%m-%d-%H-%M')
    file = 'submit%s' % datetime.datetime.strftime(datetime.datetime.now(),'%m%d')
    if os.path.exists(file):
        pass
    else:
        os.mkdir(file)
    
    
    path_csv = '{}/{}_Tr{}_t{}_Tr{}_t{}.csv'.format(file,id,f_in,t_in,f_out,t_out)
    result.to_csv(path_csv,index = False)

    ### 记录txt
    path_text = '%s/record.txt' % file
    with open(path_text,'a+') as f:
        f.write('ID:{}\tTime:{}\tPer:{}\n '.format(id,round(time.time()-t1),round(result.mean(),4)))
        f.write('Train F_in:{}\tPer:{}\t{} of {} best in train_data, p:{}\n'.format(f_in,round(result[~out_index].mean(),4),t_in,str(threshold_list_in),train_data.shape[1]))     
        f.write(str(im_features_in)+'\n')
        f.write(str(params_in)+'\n')
        
        f.write('Train F_out:{}\tPer:{}\t{} of {} best in train_data, p:{}\n '.format(f_out,round(result[out_index].mean(),4),t_out,str(threshold_list_out),train_data[out_cols].shape[1]))
        f.write(str(im_features_out)+'\n')
        f.write(str(params_out)+'\n\n')


    


    

