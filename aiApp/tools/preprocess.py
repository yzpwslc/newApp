#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:14:53 2018

@author: yzp
"""
import os
import numpy as np
import pandas as pd

try:
    from . import signalProcess
    from . import mongodb_op
    from . import txt_process
except ImportError:
    import signalProcess
    import mongodb_op
    import txt_process
    
    
class DataPreprocess(object):
    def __init__(self, **args):
        self.init_param(args)
    
    def init_param(self, args):
        self.args = args
        self.VERSION = self.args.get('version', '1_0_0')
        self.__db = mongodb_op.MongodbOp(version=self.VERSION)
        self.__txt_op = txt_process.TxtProcess()
        self._filename = self.args.get('filename')
#        self._filname = self.__txt_op.filename
        
    def origin_data(self):
        self.__txt_op.filename = self._filename
        self.df = self.__txt_op.read_txt()
        self.df.drop(['date', 'id', 'time'], axis=1, inplace=True)
        self.df = self.df.drop(self.df[~self.df.all(axis=1)].index)
        return self
    
    def data_drop(self, df, pct=0.15, drop_type='head'):
        row_num = int(df.shape[0] * pct)
        if drop_type == 'head':
            df = df.iloc[row_num : ]
        else:
            df = df.iloc[ : -row_num]
        return df
    
    def data_split(self, n=3, reset_index=True):
        sub_df = []
        for i in range(n):
            part_num = self.df.shape[0] // n
#            print('{}:{}'.format(part_num * i, part_num * (i + 1)))
            if reset_index:
                sub_df.append(self.df.iloc[part_num * i: part_num * (i + 1)].reset_index(drop=True))
        return sub_df
        

if __name__ == '__main__':
    dp = DataPreprocess(filename='/data/20181201_2/M03JS0005/WUDAO/aMIAN/FCFT3/17-20181201103408-log.txt')
    dp.origin_data()
    
        
