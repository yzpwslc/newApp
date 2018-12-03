#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:57:14 2018

@author: yzp1011
"""
import os
import re
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import mongodb_op
except ImportError:
    from . import mongodb_op


class TxtProcess(object):
    def __init__(self, **args):
        gc.collect()
        self.init_param(args)
    
    def init_param(self, args):
        self.args = args
        self._base_dir = self.args.get('base_dir', '/data')
        self._in_dir = os.path.join(self._base_dir, self.args.get('in_dir', '20181201_2'))
        self._filename = ''
        self._file_list = []
        self.__db = mongodb_op.MongodbOp()
        self._label = 1
        
        
    @property
    def filename(self):
        return self._filename
    
    @filename.setter
    def filename(self, value):
        self._filename = value
        
    @property
    def file_list(self):
        return self._file_list
    
    @file_list.setter
    def file_list(self, value):
        self._file_list = value
     
    @property    
    def label(self):
        return self._label
    
    @label.setter
    def label(self, value):
        self._label = value
        
    def file_search(self, extend='txt', basedir='', stype='all'):
        curr_dir = os.path.join(self._in_dir, basedir)
        file_lst = [os.path.join(curr_dir, f) for f in os.listdir(curr_dir) if os.path.isfile(os.path.join(curr_dir, f)) and (f.split('.')[-1].lower() == extend)]
        self.file_list += file_lst
        dir_lst = [d for d in os.listdir(curr_dir) if os.path.isdir(os.path.join(curr_dir, d))]
        [self.file_search(basedir=os.path.join(basedir, d)) for d in dir_lst]
        return self
    
    def file_db(self):
        record = self.__db.base_record
        cnc_pattern = re.compile('m[\d]{2}js[\d]+', flags=re.IGNORECASE)
        for f in self.file_list:
            record['_id'] = hash(f)
            record['a_uri'] = os.path.dirname(f)
            record['filename'] = os.path.basename(f)
            record['a_side'] = 'amian' in f.lower()
            record['youdao'] = 'youdao' in f.lower()
            record['cnc_no'] = re.findall(cnc_pattern, f)[0]
            record['test_no'] = os.path.basename(record['a_uri'])
            if record['cnc_no'] == 'M03JS0003':
                self.label = 0
            else:
                self.label = 1
            record['label'] = self.label
            self.__db.insert([record])
        return self
    
    def file_process(self, stype='all'):
        if stype == 'all':
            pass
        return self
    
    def read_txt(self):
#        print('file:{}'.format(self.filename))
        self.df = pd.read_table(self.filename, header=None, skiprows=1, delim_whitespace=True)
        self.df.columns = ['date', 'time', 'id', 'x', 'y', 'z']
#        self.df.iloc[:, 1] = self.df.iloc[:, 1].astype(np.datetime64)
        return self.df
    
    def split_data(self):
        pass
        
    
    
if __name__ == '__main__':
    tp = TxtProcess()
    tp.file_search()
    tp.file_db()
    res = tp.file_list
    tp.filename = res[5]
    print('filename:{}'.format(tp.filename))
    tp.read_txt()
    
    
    


        
        