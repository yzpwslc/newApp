#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 20:15:05 2018

@author: yzp
"""
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from sklearn.neighbors import KNeighborsClassifier
try:
    from .tools import mongodb_op, preprocess, signalProcess
except ImportError:
    from tools import mongodb_op, preprocess, signalProcess


class ClassifyModel(object):
    def __init__(self, **args):
        self.init_param(args)
        self.preprocess()
    
    def init_param(self, args):
        self.args = args
        self.__db = mongodb_op.MongodbOp()
        self.data = pd.DataFrame()
        self.t_pool = Pool(4)
        
    def preprocess(self):
#        self.data = pd.DataFrame(columns=['rpm', 'amp_ratio', 'rms', 'std', 'mean', 'mode', 'range', 'label'])
        self.data_dict = {'rpm' : [],
                          'amp_ratio' : [],
                          'rms' : [],
                          'std' : [],
                          'mean' : [],
                          'mode' : [],
                          'range' : [],
                          'label' : [],
                          }
        for label in [0, 1]:
            self.__db.query_dict = {'label' : {'$eq' : label}}
            file_list = self.__db.query()
            for f in file_list:
                file_uri = os.path.join(f['a_uri'], f['filename'])
                print(file_uri)
                dp = preprocess.DataPreprocess(filename=file_uri)
                data_pre = dp.origin_data().data_split()
                for i, df in enumerate(data_pre):
                    df = dp.data_drop(df)
                    df = dp.data_drop(df, drop_type='tail', pct=0.2)
                    self.data_dict['rpm'].append(i)
                    signal0 = signalProcess.SignalProcess(df=df)
                    self.data_dict['amp_ratio'].append(signal0.f_amp_ratio())
                    self.data_dict['rms'].append(signal0.t_v_rms())
                    self.data_dict['std'].append(signal0.t_a_std())
                    self.data_dict['mean'].append(signal0.t_a_mean())
                    self.data_dict['mode'].append(signal0.t_a_mode())
                    self.data_dict['range'].append(signal0.t_a_range())
                    self.data_dict['label'].append(label)
        self.data = pd.DataFrame().from_dict(self.data_dict) 
        
    def muti_preprocess(self):
        def helper(file_dict, label=1):
            file_uri = os.path.join(file_dict['a_uri'], file_dict['filename'])
            print(file_uri)
            dp = preprocess.DataPreprocess(filename=file_uri)
            data_pre = dp.origin_data().data_split()
            for i, df in enumerate(data_pre):
                df = dp.data_drop(df)
                df = dp.data_drop(df, drop_type='tail', pct=0.2)
                signal0 = signalProcess.SignalProcess(df=df)
                data_dict = dict()
                data_dict['rpm'] = [i]
                data_dict['amp_ratio'] = [signal0.f_amp_ratio()]
                data_dict['rms'] = [signal0.t_v_rms()]
                data_dict['std'] = [signal0.t_a_std()]
                data_dict['mean'] = [signal0.t_a_mean()]
                data_dict['mode'] = [signal0.t_a_mode()]
                data_dict['range'] = [signal0.t_a_range()]
                data_dict['label'] = label
                temp = pd.DataFrame().from_dict(data_dict)
                self.data.append(temp)
        for label in [0, 1]:
            helper_label = partial(helper, label=label)
            self.__db.query_dict = {'label' : {'$eq' : label}}
            file_list = list(self.__db.query())
            self.t_pool.map(helper_label, file_list)
                
                
    
if __name__ == '__main__':
    c_model = ClassifyModel()
