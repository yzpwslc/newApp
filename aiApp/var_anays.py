#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:57:40 2018

@author: yzp1011
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from tools import mongodb_op, preprocess, signalProcess
from itertools import combinations

VERSION = '1_0_0'

def var_distribute():
    out_dir = '/data/OUT/20181201{}'.format(VERSION)
    middle_data_file_uri = os.path.join(out_dir, 'middle/data.pkl')
    data_df = pd.read_pickle(middle_data_file_uri)
#    plt.figure(figsize=(100, 200))
    data_df.iloc[1:-1].hist()
    plt.savefig('all_var.jpg', papertype='a4')
    return data_df

def cross_var(df):
    var_name = ['rpm', 'amp_ratio', 'rms', 'std', 'mean', 'mode', 'range']
    var_tuple = combinations(var_name, 2)
    fig_num = 0
    for v in var_tuple:
        fig_num += 1
        plt.figure(fig_num)
        x, y = v
        plt.scatter(df[x], df[y], c=df['label'])
        plt.savefig('{}_{}.jpg'.format(x, y))
        
def single_var_vs_label(df):
    var_name = ['rpm', 'amp_ratio', 'rms', 'std', 'mean', 'mode', 'range']
    fig_num = 0
    for v in var_name:
        fig_num += 1
        plt.figure(fig_num)
        plt.scatter(df[v], df['label'])
        plt.savefig('{}_label.jpg'.format(v))
    
def f_envelope_spectrum():
    db = mongodb_op.MongodbOp()
    fig_num = 0
    for label in [0, 1]:
        db.query_dict = {'label' : {'$eq' : label}}
        file_dict = list(db.query())[:2]
        for f_dict in file_dict:
            file_uri = os.path.join(f_dict['a_uri'], f_dict['filename'])
            print(file_uri)
            dp = preprocess.DataPreprocess(filename=file_uri)
            data_pre = dp.origin_data().data_split()
            for i, df in enumerate(data_pre):
                df = dp.data_drop(df)
                df = dp.data_drop(df, drop_type='tail', pct=0.2)
                dp.df = df
                df_sub = dp.data_split(n=20)
                for df_temp in df_sub:
                    fig_num += 1
                    plt.figure(fig_num)
                    signal0 = signalProcess.SignalProcess(df=df_temp)
                    f, p = signal0.f_envelope_spectrum()
                    plt.plot(f, p)
                    plt.savefig('fig/{}_{}.jpg'.format(label, fig_num))
                del df_sub   
if __name__ == '__main__':
    data = var_distribute()
#    cross_var(data)
    single_var_vs_label(data)