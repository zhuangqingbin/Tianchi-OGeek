# -*- coding: utf-8 -*-

import Levenshtein
from multiprocessing import Array
import pandas as pd
import numpy as np
import itertools
from timer import timer
from reduce_mem_usage import reduce_mem_usage
import re
import time 
import jieba
#from gensim.models.keyedvectors import KeyedVectors



### 判断prefix是否在title里
def is_in_title(item):
    prefix = item["prefix"]
    title = item["title"]

    if not isinstance(prefix, str):
        prefix = "无"

    if prefix in title:
        return 1
    return 0

def in_per(item):
    title = pd.Series(list(item.title))
    prefix = pd.Series(list(item.prefix))
    return title.isin(prefix).mean()


def levenshtein(item):
    item_dict = dict()
    str1 = item["prefix"]
    str2 = item["title"]

    if not isinstance(str1, str):
        str1 = "无"
    
    ld = Levenshtein.distance(str1, str2)
    lr = Levenshtein.ratio(str1, str2)
    lj = Levenshtein.jaro(str1, str2)
    ljw = Levenshtein.jaro_winkler(str1, str2)
    item_dict['ld'] = ld
    item_dict['lr'] = lr
    item_dict['lj'] = lj
    item_dict['ljw'] = ljw
    
    return item_dict

def get_levenshtein(df):
    df["item_dict"] = df.apply(levenshtein, axis=1)
    df["ld"] = df["item_dict"].apply(lambda item: item.get("ld"))
    df["lr"] = df["item_dict"].apply(lambda item: item.get("lr"))
    df["lj"] = df["item_dict"].apply(lambda item: item.get("lj"))
    df["ljw"] = df["item_dict"].apply(lambda item: item.get("ljw"))
    df.drop(["item_dict"],axis=1,inplace=True)
    return df

def get_prefix_loc_in_title(prefix,title):
    """计算查询词prefix出现在title中的那个位置，前、后、中、没出现"""
    if prefix not in title:
        return -1
    lens = len(prefix)
    if prefix == title[:lens]:
        return 0
    elif prefix == title[-lens:]:
        return 1
    else:
        return 2


def get_rp_prefix_in_title(prefix,title,mode='char'):
    """计算title对prefix的词、字级别的召回率、精确率"""
    if mode == 'char':
        prefix = list(prefix)
        title = list(title)
    else:
        prefix = list(jieba.cut(prefix))
        title = list(jieba.cut(title))  
    len_title = len(title)
    len_prefix = len(prefix)
    len_comm_xx = len(set(prefix) & set(title))
    
    recall = len_comm_xx / (len_prefix + 0.01)
    precision = len_comm_xx / (len_title + 0.01)
    acc = len_comm_xx / (len_title + len_prefix - len_comm_xx)
    return [recall,precision,acc]

def get_ngram_rp_prefix_in_title(prefix,title,mode='char'):
    """计算title对prefix的词、字级别的召回率、精确率（1-2gram）"""
    if mode == 'char':
        prefix = list(prefix)
        title = list(title)
    else:
        prefix = list(jieba.cut(prefix))
        title = list(jieba.cut(title))
    prefix_2gram = []
    for i in range(len(prefix)-1):
        prefix_2gram.append(prefix[i]+prefix[i+1])
    prefix.extend(prefix_2gram)
    
    title_2gram = []
    for i in range(len(title)-1):
        title_2gram.append(title[i]+title[i+1])
    title.extend(title_2gram)
    
    len_title = len(title)
    len_prefix = len(prefix)
    len_comm_xx = len(set(prefix) & set(title))
    
    recall = len_comm_xx / (len_prefix + 0.01)
    precision = len_comm_xx / (len_title + 0.01)
    acc = len_comm_xx / (len_title + len_prefix - len_comm_xx)
    return [recall,precision,acc]
    
def get_rp_query_in_title(query,title,mode='char'):
    """计算title对query中概率最大句子的词、字级别的召回率、精确率"""
    if len(query) == 0:
        return [-1,-1,-1]
    query = sorted(query.items(),key = lambda x:x[1],reverse = True)
    query_str = query[0][0]
    if float(query[0][1]) < 0.1:
        return [0,0,0]
    else:
        return get_rp_prefix_in_title(query_str,title,mode=mode)
        
def get_ngram_rp_query_in_title(query,title,mode='char'):
    """计算title对query中概率最大句子的词、字级别的召回率、精确率（1-2gram）"""
    if len(query) == 0:
        return [-1,-1,-1]
    query = sorted(query.items(),key = lambda x:x[1],reverse = True)
    query_str = query[0][0]
    if float(query[0][1]) < 0.1:
        return [0,0,0]
    else:
        return get_ngram_rp_prefix_in_title(query_str,title,mode=mode)
    

def AddTextFeatures(data_):
    print("-"*50)
    print('Begining Text Feature Add......')
    t1 = time.time()

    data = data_.copy()

    
    ### 计算是否在title
    with timer("计算是否在title"):
        data['in_title'] = data.apply(is_in_title, axis=1)
        

    
    ### 计算Levenshtein
    with timer("计算Levenshtein"):
        data = get_levenshtein(data)
    
    ### 计算查询词prefix出现在title中的位置
    with timer("计算查询词prefix出现在title中的位置"):
        data['prefix_loc'] = data.apply(lambda x : get_prefix_loc_in_title(x['prefix'],x['title']), axis=1)
    
    
    ### 计算title对prefix的词、字级别的召回率、精确率
    with timer("计算title对prefix的词、字级别的召回率、精确率"):    
        char_level_prefix = data.apply(lambda x : get_rp_prefix_in_title(x['prefix'],x['title'],mode='char'), axis=1)
        char_level_prefix = [kk for kk in char_level_prefix]
        char_level_prefix = np.array(char_level_prefix)
        data['prefix_t_recall_char'] = char_level_prefix[:,0].tolist()
        data['prefix_t_precision_char'] = char_level_prefix[:,1].tolist()
        data['prefix_t_acc_char'] = char_level_prefix[:,2].tolist()
    
        word_level_prefix = data.apply(lambda x : get_rp_prefix_in_title(x['prefix'],x['title'],mode='word'), axis=1)
        word_level_prefix = [kk for kk in word_level_prefix]
        word_level_prefix = np.array(word_level_prefix)
        data['prefix_t_recall_word'] = word_level_prefix[:,0].tolist()
        data['prefix_t_precision_word'] = word_level_prefix[:,1].tolist()
        data['prefix_t_acc_word'] = word_level_prefix[:,2].tolist()
    
    
    ### 计算title对prefix的词、字级别的召回率、精确率（1-2gram）
    with timer("计算title对prefix的词、字级别的召回率、精确率（1-2gram）"):     
        char_ngram_level_prefix = data.apply(lambda x : get_ngram_rp_prefix_in_title(x['prefix'],x['title'],mode='char'), axis=1)
        char_ngram_level_prefix = [kk for kk in char_ngram_level_prefix]
        char_ngram_level_prefix = np.array(char_ngram_level_prefix)
        data['prefix_t_recall_char_ngram'] = char_ngram_level_prefix[:,0].tolist()
        data['prefix_t_precision_char_ngram'] = char_ngram_level_prefix[:,1].tolist()
        data['prefix_t_acc_char_ngram'] = char_ngram_level_prefix[:,2].tolist()
        
        word_ngram_level_prefix = data.apply(lambda x : get_ngram_rp_prefix_in_title(x['prefix'],x['title'],mode='word'), axis=1)
        word_ngram_level_prefix = [kk for kk in word_ngram_level_prefix]
        word_ngram_level_prefix = np.array(word_ngram_level_prefix)
        data['prefix_t_recall_word_ngram'] = word_ngram_level_prefix[:,0].tolist()
        data['prefix_t_precision_word_ngram'] = word_ngram_level_prefix[:,1].tolist()
        data['prefix_t_acc_word_ngram'] = word_ngram_level_prefix[:,2].tolist()
        
        
    ### 计算title对query中概率最大句子的词、字级别的召回率、精确率
    with timer("计算title对query中概率最大句子的词、字级别的召回率、精确率）"):     
        char_level_query = data.apply(lambda x : get_rp_query_in_title(x['query_prediction'],x['title'],mode='char'), axis=1)
        char_level_query = [kk for kk in char_level_query]
        char_level_query = np.array(char_level_query)
        data['query_t_recall_char'] = char_level_query[:,0].tolist()
        data['query_t_precision_char'] = char_level_query[:,1].tolist()
        data['query_t_acc_char'] = char_level_query[:,2].tolist()
    
        word_level_query = data.apply(lambda x : get_rp_query_in_title(x['query_prediction'],x['title'],mode='word'), axis=1)
        word_level_query = [kk for kk in word_level_query]
        word_level_query = np.array(word_level_query)
        data['query_t_recall_word'] = word_level_query[:,0].tolist()
        data['query_t_precision_word'] = word_level_query[:,1].tolist()
        data['query_t_acc_word'] = word_level_query[:,2].tolist()
    

    ### 计算title对query中概率最大句子的词、字级别的召回率、精确率（1-2gram）
    with timer("计算title对query中概率最大句子的词、字级别的召回率、精确率（1-2gram）"):     
        char_ngram_level_query = data.apply(lambda x : get_ngram_rp_query_in_title(x['query_prediction'],x['title'],mode='char'), axis=1)
        char_ngram_level_query = [kk for kk in char_ngram_level_query]
        char_ngram_level_query = np.array(char_ngram_level_query)
        data['query_t_recall_char_ngram'] = char_ngram_level_query[:,0].tolist()
        data['query_t_precision_char_ngram'] = char_ngram_level_query[:,1].tolist()
        data['query_t_acc_char_ngram'] = char_ngram_level_query[:,2].tolist()
    
        word_ngram_level_query = data.apply(lambda x : get_ngram_rp_query_in_title(x['query_prediction'],x['title'],mode='word'), axis=1)
        word_ngram_level_query = [kk for kk in word_ngram_level_query]
        word_ngram_level_query = np.array(word_ngram_level_query)
        data['query_t_recall_word_ngram'] = word_ngram_level_query[:,0].tolist()
        data['query_t_precision_word_ngram'] = word_ngram_level_query[:,1].tolist()
        data['query_t_acc_word_ngram'] = word_ngram_level_query[:,2].tolist()  
    
    
    ### 压缩数据内存
    with timer("压缩数据内存"):
        data = reduce_mem_usage(data)
    
    print('-'*50)
    print("{} - done in {:.0f}s".format('Add Text Feature', time.time() - t1))
    
    return data
