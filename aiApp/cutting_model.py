#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 08:35:24 2018

@author: yzp1011
"""
import os
import eli5
import xgboost
import gc
import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV

try:
    from .tools import preprocess
    from .tools import signalProcess
    from .tools import mongodb_op
except ImportError:
    from tools import preprocess
    from tools import signalProcess
    from tools import mongodb_op


class CuttingToolModel(object):
    def __init__(self, **args):
        self.init_param(args)
    
    def init_param(self, args):
        self.sig = signalProcess.SignalProcess(df=pd.DataFrame())
        self.dp = preprocess.DataPreprocess()
        self.mdb = mongodb_op.MongodbOp()
        self.segment = 'train'
        
    def renference(self):
        self.dp.query_dict = {'test_no': '13931#'}
        df = self.dp.data_concat()
        self.sig.df = df
        post = self.sig.t_data_process_by_col()
        del df
        self.sig.df = pd.DataFrame()
        post.dropna(axis=0, inplace=True)
        post_30s = post.rolling(20).mean()
        post_1s = post.shift(1)
        post_1s.dropna(axis=0, inplace=True)
        post = post.merge(post_1s, left_index=True, right_index=True, how='inner', suffixes=['', 'last_1_s'])
        post = post.merge(post_30s, left_index=True, right_index=True, how='inner', suffixes=['', 'last_30_s'])
        post.dropna(axis=0, inplace=True)
        res_cut, q_df = self.dp.data_cut_bins(post, bins=3)
        res_cut = res_cut.astype(str)
        dummies_res = pd.get_dummies(res_cut, columns=res_cut.columns)
        del res_cut
        dummies_res['label'] = 1
        dummies_res.loc[ : '2018-12-03 17:27:00', 'label'] = 0
        x_train, x_test, y_train, y_test = train_test_split(dummies_res.drop(['label'], axis=1),dummies_res['label'], train_size=0.9, shuffle=True)
        dmatrix_train = xgboost.DMatrix(x_train, y_train)
        xgb = xgboost.XGBClassifier()
        param_dict = {
                'learning_rate' : np.linspace(0.03, 0.3, 4),
                'n_estimators' : [50, 75, 100, 150],
                'max_depth' :  [4, 5, 6, 7],
                }
        gs = GridSearchCV(xgb, param_dict, cv=3, scoring='roc_auc')
        gs.fit(x_train.values, y_train.astype(np.int8).values)
        print('score:{}'.format(gs.best_score_))
#        gs.best_estimator_.train(dmatrix_train)
        return x_train, y_train
    
    
if __name__ == '__main__':
    ctm = CuttingToolModel()
    d, q = ctm.renference()
        
        
        
