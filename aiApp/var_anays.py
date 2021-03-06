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
from scipy import signal

VERSION = '1_0_1'

def var_distribute():
    out_dir = '/data/OUT/20181201{}'.format(VERSION)
    middle_data_file_uri = os.path.join(out_dir, 'train_middle/data.pkl')
    data_df = pd.read_pickle(middle_data_file_uri)
#    plt.figure(figsize=(100, 200))
    data_df.iloc[1:-1].hist()
    plt.savefig('all_var.jpg', papertype='a4')
    print(data_df.drop(['label'], axis=1).corr())
    return data_df

def origin_distribute():
    db = mongodb_op.MongodbOp()
    db._query_dict = {'label' : {'$eq' : 0}}
    file_dict = db.query()
    temp = []
    for f_dict in file_dict:
        
        file = os.path.join(f_dict.get('a_uri'), f_dict.get('filename'))
        print(file)
        dp = preprocess.DataPreprocess(filename=file)
        dp.origin_data()
        dp.df = dp.df.drop(['x', 'y'], axis=1)
        df_rpm_1, _, _ = dp.data_split()
        df_rpm_1 = dp.data_drop(df_rpm_1,)
        df_rpm_1 = dp.data_drop(df_rpm_1, drop_type='tail')
        df_rpm_1.columns = [file]
        temp.append(df_rpm_1)
    return pd.concat(temp, sort=False, axis=1)
        
def muti_proc_op():
    pass

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
                
def abnormal_zero_anasys():
    figure_no = 0
    db = mongodb_op.MongodbOp()
    db._query_dict = {'label' : {'$eq' : 1}}
    file_dict = db.query()
    temp = []
    for f_dict in file_dict:
        figure_no += 1
        plt.figure(figure_no)
        file = os.path.join(f_dict.get('a_uri'), f_dict.get('filename'))
        print(file)
        dp = preprocess.DataPreprocess(filename=file)
        dp.origin_data(filter_zero=False)
        dp.df.loc[dp.df.all(axis=1), 'z'] = 1
        f, p = signal.welch(dp.df.z, 4000, scaling='spectrum')
        plt.plot(f, p)
    return f, p
    

if __name__ == '__main__':
#    data = var_distribute()
#    cross_var(data)
#    single_var_vs_label(data)
#    df = origin_distribute()
    f, p = abnormal_zero_anasys()