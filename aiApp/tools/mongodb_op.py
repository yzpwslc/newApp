#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:48:44 2018

@author: yzp
"""
import os
from pymongo import MongoClient


class MongodbOp(object):
    def __init__(self, **args):
        self.init_param(args)
        self.connect()
    
    def init_param(self, args):
        self.__args = args
        self.VERSION = args.get('version', '1_0_0')
        self.__host = 'localhost'
        self.__port = 27017
        self.__db_name = 'test'
        self.__collections_name = 'file_info_{}'.format(self.VERSION)
        self._query_dict = dict()
        self.base_record = {'_id' : '',
                            'filename' : '',
                            'a_uri' : '',
                            'cnc_no' : '',
                            'youdao' : True,
                            'a_side' : True,
                            'test_no' : 'FCFT1',
                            'extend' : '',
                            'label' : 1,
                            }
        
    def connect(self):
        host = self.__args.get('host', self.__host)
        port = self.__args.get('port', self.__port)
        client = MongoClient(host, port)
        self.__db = client[self.__db_name]
        return self
    
    @property
    def db_name(self):
        return self.__db_name
    
    @db_name.setter    
    def db_name(self, value):
        if value != '':
            self.__db_name = value
            
    @property
    def collections_name(self):
        return self.__collections_name
    
    @collections_name.setter    
    def collections_name(self, value):
        if value != '':
            self.__collections_name = value
            
    @property
    def query_dict(self):
        return self._query_dict
    
    @query_dict.setter    
    def query_dict(self, value):
        if value != '':
            self._query_dict = dict()
            self._query_dict = value    
        
    def count(self):
        res = 0
        res = self.__db[self.__collections_name].count_documents(self.query_dict)
        return res
    
    def query(self, stype='all'):
        res = None
        if stype == 'all':
            res = self.__db[self.__collections_name].find(self.query_dict)
        else:
            res = self.__db[self.__collections_name].find_one(self.query_dict)
        return res
            
    def insert(self, record):
        if len(record) > 1:
            self.__db[self.__collections_name].insert_many(record)
        else:
            self.__db[self.__collections_name].insert_one(record[0])
        return self
            
    def delete(self, stype='all'):
        if stype == 'all':
            self.__db[self.__collections_name].delete_many(self.query_dict)
        else:
            self.__db[self.__collections_name].delete_one(self.query_dict)
        return self
    
    def distinct(self, key):
        return self.__db[self.__collections_name].distinct(key, filter=self.query_dict)
            
            
if __name__ == '__main__':
    mdb = MongodbOp()
            
    