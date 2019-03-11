#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:47:47 2018

@author: jimmy
"""
import numpy as np
import pandas as pd
import time
from contextlib import contextmanager



@contextmanager
def timer(title):
    """
    Record system time.

    Parameters
    ----------
    title : str
        Timer name.
    
    Examples
    --------
    >>> def test():
            time.sleep(5)

    >>> with timer("Function test:"):
            test()
    --------------------------------------------------
    Function test: - done in 5s
    """
    t0 = time.time()
    print("-"*50)
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    #print("-"*20+'\n')
    


if __name__=="__main__":
    def test():
        time.sleep(5)
    with timer("Function test:"):
        test()

    data = pd.DataFrame({'x':['a','a','b','c'],'y':[1,2,3,4]})
    data.head()
    df,cols = one_hot_encoder(data, nan_as_category = True)
    df.head()
    cols
