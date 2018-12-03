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
#        self._filname = self.__txt_op.filename
        
    def origin_data(self):
        file_lst = self.__db.query()
        records = list(file_lst)
        record = np.random.choice(records, 1)[0]
        self.__txt_op.filename = os.path.join(record['a_uri'], record['filename'])
        self.df = self.__txt_op.read_txt()
        return self.df.index.values, self.df.z.values
    
    def data_split(self, n=3):
        sub_df = []
        for i in range(n):
            part_num = self.df.shape[0] // n
#            print('{}:{}'.format(part_num * i, part_num * (i + 1)))
            sub_df.append(self.df.iloc[part_num * i: part_num * (i + 1)])
        return sub_df
        

if __name__ == '__main__':
    dp = DataPreprocess()
    dp.origin_data()
    
        
