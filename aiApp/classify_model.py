#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 20:15:05 2018

@author: yzp
"""
import os
import gc
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
try:
    from .tools import mongodb_op, preprocess, signalProcess
except ImportError:
    from tools import mongodb_op, preprocess, signalProcess, plot_confusion_matrix
    
VERSION = '1_0_0'

def helper(file_dict, out_dir, label=1):
    gc.collect()
    file_uri = os.path.join(file_dict['a_uri'], file_dict['filename'])
    print(file_uri)
    dp = preprocess.DataPreprocess(filename=file_uri)
    data_pre = dp.origin_data().data_split()
    temp_lst = []
    for i, df in enumerate(data_pre):
        df = dp.data_drop(df)
        df = dp.data_drop(df, drop_type='tail', pct=0.2)
        dp.df = df
        df_sub = dp.data_split(n=20)
        for df_temp in df_sub:
#            df_temp = df.sample(frac=0.4)
            signal0 = signalProcess.SignalProcess(df=df_temp)
            print('{} signal process'.format(file_uri))
            data_dict = dict()
            data_dict['rpm'] = [i]
            data_dict['amp_ratio'] = [signal0.f_amp_ratio()]
            data_dict['rms'] = [signal0.t_v_rms()]
            data_dict['std'] = [signal0.t_a_std()]
            data_dict['mean'] = [signal0.t_a_mean()]
            data_dict['mode'] = [signal0.t_a_mode()]
            data_dict['range'] = [signal0.t_a_range()]
            data_dict['label'] = label
            temp_lst.append(pd.DataFrame().from_dict(data_dict))
            del df_temp
#        all_df.concat(temp, sort=False, ignore_index=True, copy=False)
#        all_df.to_pickle('{}.pkl'.format(os.path.basename(file_uri)))
#        del temp
        del df_sub
        del df
    del data_pre
    pd.concat(temp_lst, ignore_index=True).to_pickle(os.path.join(out_dir, '{}.pkl'.format(file_dict['filename'])))
    del temp_lst
    print('--process finished')
    return 1
    

class ClassifyModel(object):
    def __init__(self, **args):
        self.init_param(args)
#        self.preprocess()

    def init_param(self, args):
        self.args = args
        self.stage = 'train'
        self.out_dir = '/data/OUT/20181201{}'.format(VERSION)
        self.__db = mongodb_op.MongodbOp()
        self.data = pd.DataFrame(columns=['rpm', 'amp_ratio', 'rms', 'std', 'mean', 'mode', 'range', 'label'])
        self.t_pool = Pool(6)
        self.x_predict = pd.DataFrame()
        
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
            
    def get_data(self):
#        def helper(file_dict, label=1):
#            print(file_dict)
        if os.path.exists(os.path.join(self.out_dir, '{}_middle'.format(self.stage), 'data.pkl')):
            self.data = pd.read_pickle(os.path.join(self.out_dir, '{}_middle'.format(self.stage), 'data.pkl'))
        else:
            out_dir = os.path.join(self.out_dir, '{}_data'.format(self.stage))
            print(out_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            if self.stage == 'train':
                for label in [0, 1]:
                    helper_label = partial(helper, out_dir=out_dir, label=label)
                    self.__db.query_dict = {'label' : {'$eq' : label}}
                    file_list = list(self.__db.query())
        #            print(file_list)
                    self.t_pool.map(helper_label, file_list)
            else:
                helper_label = partial(helper, out_dir=out_dir, label=3)
                self.__db.query_dict = {'label' : {'$eq' : 3}}
                file_list = list(self.__db.query())
        #            print(file_list)
                self.t_pool.map(helper_label, file_list)            
            print('finished')
            self.contact_data()        
#        self.data.to_pickle('data.pkl')
        
    def refrence(self):
        model_dir = '/data/OUT/20181201{}/model'.format(VERSION)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.stage == 'train':
            self.model = SVC()
            search_dict = {'C': [0.2, 0.5, 1.0], 
                    'kernel': ['linear', 'rbf', 'sigmoid'], 
                    'gamma': [0.1, 0.2, 0.3],
                    }
            gs = GridSearchCV(self.model, cv=3, param_grid=search_dict, scoring='roc_auc')
            gs.fit(self.x_train.values, self.y_train.values)
            print('best score:{}'.format(gs.best_score_))
            print('best param:{}'.format(gs.best_params_))
            self.model = gs.best_estimator_
            with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
                pickle.dump(self.model, f)
        else:
            self.model = pickle.load(os.path.join(model_dir, 'model.pkl'))
            self.model.predict(self.x_predict)
            
    def result_analysis(self):
        if self.stage == 'train':
            y_true = self.y_test.values
            cf_matrix = confusion_matrix(y_true, self.model.predict(self.x_test.values))
            plot_confusion_matrix.plot_confusion_matrix(cf_matrix, classes=[0, 1])
        else:
            y_true = np.array([0] * self.data.shape[0])
            print('y_true:{}'.format(len(y_true)))
            cf_matrix = confusion_matrix(y_true, self.model.predict(self.data.drop(['label'], axis=1).values))
            plt.figure(2)
            plot_confusion_matrix.plot_confusion_matrix(cf_matrix, classes=[0, 1])
        
    
    def contact_data(self):
        f_list = [os.path.join(self.out_dir, '{}_data'.format(self.stage), f) for f in os.listdir(os.path.join(self.out_dir, '{}_data'.format(self.stage))) if f.endswith('pkl')]
        for f in f_list:
            self.data = self.data.append(pd.read_pickle(f), ignore_index=True)
        out_dir = os.path.join(self.out_dir, '{}_middle'.format(self.stage))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.data.to_pickle(os.path.join(out_dir, 'data.pkl'))
        
    def repair_data(self, cols=['amp_ratio', 'mean'], func='log'):
        op = {'log': np.log, 
              'exp': np.exp
              }
        for c in cols:
            self.data.loc[:, c] = self.data[c].map(op.get(func))
        return self
    
    def model_pipline(self):
        self.get_data()
        self.repair_data()
        self.data.loc[:, ['rpm', 'label']] = self.data[['rpm', 'label']].astype(np.int8)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data.drop(['label'], axis=1), self.data['label'], stratify=self.data['label'], shuffle=True, test_size=0.2)
        self.refrence()
        self.result_analysis()
        self.stage = 'test'
        self.data = pd.DataFrame(columns=['rpm', 'amp_ratio', 'rms', 'std', 'mean', 'mode', 'range', 'label'])
        self.get_data()
        self.repair_data()
        self.data.loc[:, ['rpm', 'label']] = self.data[['rpm', 'label']].astype(np.int8)     
        self.result_analysis()


if __name__ == '__main__':
    c_model = ClassifyModel()
    c_model.model_pipline()
