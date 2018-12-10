#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:14:53 2018

@author: yzp
"""
import os
import gc
import numpy as np
import pandas as pd
import datetime
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
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
        self._filename = self.args.get('filename', '')
        self._query_dict = {'test_no': '13931#'}
#        self._filname = self.__txt_op.filename
    
    @property
    def query_dict(self):
        return self._query_dict
    
    @query_dict.setter
    def query_dict(self, value):
        if value != '':
            self._query_dict = value
        
    def origin_data(self, filter_zero=True, drop_cols=['date', 'id', 'time']):
        self.__txt_op.filename = self._filename
        self.df = self.__txt_op.read_txt()
        self.df.drop(drop_cols, axis=1, inplace=True)
        if filter_zero:
            self.df = self.df.drop(self.df[~self.df.any(axis=1)].index)
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
    
    def data_concat(self, cols=['date', 'time', 'id', 'x', 'y', 'z']):
        out_file = '/data/OUT/QIE/middle/df_all.pkl'
        try:
            r_df = pd.read_pickle(out_file)
        except:
            self.__db.collections_name = 'cut_data_file_{}'.format('1_0_0')
            r_df = pd.DataFrame(columns = cols)
            
            self.__db.query_dict = self.query_dict
    #        print(list(self.__db.query()))
            for f_dict in self.__db.query():
                file = os.path.join(f_dict.get('a_uri'), f_dict.get('filename'))
                print(file)
                self.__txt_op.filename = file
                self.df = self.__txt_op.read_txt()
    #            self.df['time'] = self.df['date'] + ' ' + self.df['time']
    #            self.df.loc['time'] = self.df['time'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                r_df = r_df.append(self.df)
            r_df['time_stamp'] = r_df['date'] + ' ' + r_df['time']
            r_df.drop(['date', 'time', 'id'], inplace=True, axis=1)
            r_df.loc[:, 'time_stamp'] = r_df['time_stamp'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            r_df.set_index('time_stamp', inplace=True)
            r_df.dropna(axis=0)
            if not os.path.exists(os.path.dirname(out_file)):
                os.makedirs(os.path.dirname(out_file))
            r_df.to_pickle(out_file)
        return r_df
    
#    def data_process_by_col(self, df, resample_period='5S', is_save=True):
#        df_post = df.resample(resample_period).agg({'mean': np.mean, 
##                             'mode': lambda x: x.mode().mean(), 
#                             'std': np.std, 
#                             'range': lambda x: x.max() - x.min(), 
##                             'skew': 'skew', 
#                             'kurtosis': kurtosis, 
#                             'alpha': lambda x: (x ** 3).mean(), 
##                             'beta': lambda x: (x ** 4).mean(),
#                             })
#        return df_post
    
    def data_plot(self, post):
        for i, c in enumerate(post.columns):
            plt.figure(figsize=(12, 9))
            plt.subplot(len(post.columns), 1, i + 1)
            post[c].plot()
            plt.show()
            
    def data_cut_bins(self, df, q_df=None, bins=3, cols=[]):
        if q_df is None:
            q_df = pd.DataFrame()
            q_list = []
            if len(cols) == 0:
                columns = df.columns
            else:
                columns = cols
            for c in columns:
                print('process {}'.format(c))
                temp, q_bins = pd.qcut(df[c], bins, labels=False, retbins=True)
                df.loc[:, c] = temp
                q_list.append(pd.DataFrame(q_bins))
#                self.list = q_list
#                self.df = df
            q_df = pd.concat(q_list, axis=1)
                
        else:
            if len(cols) == 0:
                for c in df.columns:
                    cut_bins = q_df[c].values
                    temp = pd.cut(df[c], bins=cut_bins, labels=False)
                    df.loc[:, c] = temp
        return df, q_df
    
                    
                    
if __name__ == '__main__':
    dp = DataPreprocess(filename='/data/20181201_2/M03JS0003/WUDAO/bMIAN/FCFT2/17-20181201110318-log.txt')
#    dp.origin_data()
    res = dp.data_concat()
#    post = dp.data_process_by_col(res)
    del res
    gc.collect()

    
    
        
