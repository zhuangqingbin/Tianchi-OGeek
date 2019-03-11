#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:57:40 2018

@author: jimmy
"""


threshold_list_in = [0.38,0.39,0.4,0.405,0.41,0.415,0.42,0.425,0.43,0.44,0.45]
threshold_list_out = [0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45]



params_in = {
'boosting_type': 'gbdt','objective': 'binary','metric': 'binary_logloss',
'num_leaves': 30,'learning_rate': 0.01,'feature_fraction': 0.8,
'bagging_fraction': 0.8,'bagging_freq': 8,'verbose': 1
}
params_out = {
'boosting_type': 'gbdt','objective': 'binary','metric': 'binary_logloss',
'num_leaves': 30,'learning_rate': 0.05,'feature_fraction': 0.8,
'bagging_fraction': 0.8,'bagging_freq': 5,'verbose': 1,
'early_stopping_rounds':100,'num_boost_round':30000
}



















