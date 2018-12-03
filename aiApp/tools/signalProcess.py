#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 08:23:09 2018

@author: yzp1011
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import tdms_op
except:
    from . import tdms_op
from scipy import signal, fftpack


class SignalProcess(object):
    def __init__(self, **args):
        self.init_param(args)
    
    def init_param(self, args):
        self.args = args
        if args.get('df', None) == None:
            self.data_file = args['data_file']
            self.in_dir = args.get('in_dir', '')
            self.dp = tdms_op.TdmsProcess(filename=self.data_file, in_dir=self.in_dir)
            self._df = self.dp.to_df()
        else:
            self._df = args['df']
        self.sample_freq = 4000
        self.time_step = 1 / self.sample_freq
            
    @property       
    def df(self):
        return self._df
    
    @df.setter
    def df(self, dataframe):
        if dataframe is not None:
            self._df = dataframe
            
    def f_envelope_spectrum(self):
        analytic_signal = signal.hilbert(self.df.iloc[:, 2])
        amplitude_envelope = np.abs(analytic_signal)
        fs = 4000
#        freq = fftpack.fftfreq(analytic_signal.size, d=1 / fs)
#        sig_fft = [abs(fp) for fp in fftpack.fft(self.df.iloc[:, 2])]
#        plt.plot(freq, sig_fft)
        f, p = signal.welch(amplitude_envelope, self.sample_freq, scaling='spectrum')
#        peaks, _ = signal.find_peaks(p, distance=100)
#        plt.plot(f[:-1], np.diff(p))
##        plt.plot(peaks, p[peaks], 'x')
#        plt.show()
        return f, p
    
    def find_peaks(self, arr, window_length, ratio):
        diff_arr = np.diff(arr[1])
        
    def t_v_rms(self):
        self.df = self.df.rolling(self.sample_freq).mean()
        return self.df.map(lambda x : x ** 2).mean()
    
if __name__ == '__main__':
    in_dir = '/data/20180915'
    file_list = set([f for f in os.listdir(in_dir) if f.endswith('tdms')])
    spindle_file_list = set([f for f in file_list if 'spindle' in f])
    print('all:{} , spindle:{}'.format(len(file_list), len(spindle_file_list)))
#    print(spindle_file_list)
    file = spindle_file_list.pop()
    print(file)
    signalprocess = SignalProcess(data_file='lh_spindle_health_assessment_20180915_004002.tdms', in_dir=in_dir)
    res = signalprocess.f_envelope_spectrum()
    
        


