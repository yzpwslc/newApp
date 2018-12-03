#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:23:52 2018

@author: yzp1011
"""
import os
import sys
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from datetime import datetime

class TdmsProcess(object):
    def __init__(self, filename='', in_dir='./IN', op_type='to_pickle', extend=''):
        self.file_name = filename
        self.extend = extend
        self.op = op_type
        self.init_param()
        self.file = os.path.join(in_dir, filename)
        self.read_tdms()
#        if op_type != 'to_df':
#            self.to_df()
            
    def init_param(self):
        self.df = pd.DataFrame()
        self.out_dir = './OUT'
        self.op_dict = {'to_pickle': self.to_pickle, 
                   'to_df': self.to_df, 
                   'to_h5': self.to_h5
                   }
    
    def read_tdms(self):
        self.file_object = TdmsFile(self.file)
        return self
        
    def to_pickle(self):
        time = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S%f')
        out_file_name = os.path.join(self.out_dir, self.extend, '{}.pkl'.format(self.file_name.split('.')[0]))
        self.to_df().to_pickle(out_file_name)
        return out_file_name
              
    def to_h5(self):
#        time = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S%f')
        
        out_file_name = os.path.join(self.out_dir, self.extend, '{}.h5'.format(self.file_name.split('.')[0]))        
        with pd.HDFStore(out_file_name) as h5:
            h5['data'] = self.to_df()
        return out_file_name
    
    def to_df(self):
        self.df = self.file_object.as_dataframe()
        return self.df
    
if __name__ == '__main__':
    in_dir = '/data/20180915'
    file_list = set([f for f in os.listdir(in_dir) if f.endswith('tdms')])
    spindle_file_list = set([f for f in file_list if 'spindle' in f])
    print('all:{} , spindle:{}'.format(len(file_list), len(spindle_file_list)))
    print(spindle_file_list)
    file = spindle_file_list.pop()
    print(file)
    tdms = TdmsProcess(filename=file, in_dir=in_dir)
    
    
    