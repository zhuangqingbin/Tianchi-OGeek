#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 18:33:07 2018

@author: jimmy
"""
import numpy as np
import pandas as pd

def reduce_mem_usage(data, verbose = True):
    """
    Reduce the memory usage of data.

    Parameters
    ----------
    data : DataFrame
        The data which memory usage will be reduced.
    verbose : bool, default True
        if True, the reduced memory usage information will be printed.
    
    Returns
    -------
    The mem-usage-reduced data.
    
    Examples
    --------
    >>> data = pandas.DataFrame({'x':['a','b']*3000,
        'y':[1.4,1.3,4.5]*2000,'z':[True,False]*3000})
    >>> data.head()
       x         y      z
    0  a  1.400391   True
    1  b  1.299805  False
    2  a  4.500000   True
    3  b  1.400391  False
    4  a  1.299805   True

    >>> new_data = reduce_mem_usage(data)
    Memory usage of dataframe: 0.10 MB
    Memory usage after optimization: 0.06 MB
    Decreased by 35.3%
    
    >>> new_data.head()
       x         y      z
    0  a  1.400391   True
    1  b  1.299805  False
    2  a  4.500000   True
    3  b  1.400391  False
    4  a  1.299805   True
    
    """
    start_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage of dataframe: {:.2f} MB'.format(start_mem))
    types = [np.int,np.int8,np.int16,np.int32,np.int64]
    types.extend([np.float,np.float16,np.float32,np.float64])
    for col in data.columns:
        col_type = data[col].dtype
        
        if col_type in types:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return data
