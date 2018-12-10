#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 08:23:09 2018

@author: yzp1011
"""
import os
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from PyEMD import EEMD
from scipy.stats import kurtosis
try:
    import tdms_op
    import preprocess
    import mongodb_op
except:
    from . import tdms_op
    from . import preprocess
from scipy import signal, fftpack


class SignalProcess(object):
    def __init__(self, **args):
        self.init_param(args)
    
    def init_param(self, args):
        self.args = args
#        if args.get('df', None) is None:
#            self.data_file = args['data_file']
#            self.in_dir = args.get('in_dir', '')
#            self.dp = tdms_op.TdmsProcess(filename=self.data_file, in_dir=self.in_dir)
#            self._df = self.dp.to_df()
#        else:
#            self._df = args['df']
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
        analytic_signal = signal.hilbert(self.df.z)
        amplitude_envelope = np.abs(analytic_signal)
#        fs = 4000
#        f = fftpack.fftfreq(analytic_signal.size, d=1 / fs)
#        p = [abs(fp) for fp in fftpack.fft(self.df.iloc[:, 2])]
#        f = f[f.size // 10: f.size // 2]
#        p = p[len(p) // 10: len(p) // 2]
#        plt.plot(freq, sig_fft)
        f, p = signal.welch(amplitude_envelope, self.sample_freq, scaling='spectrum')
        return f, p
    
    def f_fft(self):
#        f = fftpack.fftfreq(self.df.z.size, d=1 / self.sample_freq)
#        p = [abs(fp) for fp in fftpack.fft(self.df.z)]
        f, p = signal.welch(self.df.z, self.sample_freq, scaling='spectrum')
        return f, p
        
    
    def f_amp_ratio(self, split_freq=750):
        f, p = self.f_envelope_spectrum()
        ratio = p[ : f[f <= split_freq].size].max() / p[f[f > split_freq].size : ].max()
        return ratio
    
    def f_eemd(self, cols='z'):
        t = (self.df.index * self.time_step).values
        eemd = EEMD()
        emd = eemd.EMD
        emd.extrema_detection = 'parabol'
        eIMFs = eemd.eemd(self.df[cols].values, t)
        nIMFs = eIMFs.shape[0]
        plt.figure(figsize=(12, 9))
        plt.subplot(nIMFs+1, 1, 1)
        plt.plot(t, S, 'r')
        for n in range(nIMFs):
            plt.subplot(nIMFs+1, 1, n+2)
            plt.plot(t, eIMFs[n], 'g')
            plt.ylabel("eIMF %i" %(n+1))
            plt.locator_params(axis='y', nbins=5)
        plt.tight_layout()
        plt.savefig('eemd_example', dpi=120)
        plt.show()
        
    def t_v_rms(self, col='z', r_type='a'):
        if r_type == 'v':
            self.df.loc[:, col] = (self.df[col] * self.time_step - (self.df[col] * self.time_step).mean()).cumsum() 
#        return self.df.map(lambda x : x ** 2).rolling(self.sample_freq).mean().dropna().reset_index(drop=True)
        return self.df[col].map(lambda x : x ** 2).agg('mean')
    
    def t_a_std(self, col='z'):
        return self.df[col].agg('std')
    
    def t_a_mean(self, col='z'):
        return self.df[col].agg('mean')
    
    def t_a_mode(self, col='z'):
        return self.df[col].agg('mode').values.mean()
    
    def t_a_range(self, col='z'):
        return self.df[col].max() - self.df[col].min()
    
    def t_data_process_by_col(self, resample_period='1S', is_save=True):
        print('process data by col')
        out_file = '/data/OUT/QIE/middle/df_s.pkl'
        if os.path.exists(out_file):
            return pd.read_pickle(out_file)

        df_post = self.df.resample(resample_period).agg({'mean': np.mean, 
#                             'mode': lambda x: x.mode().mean(), 
                             'std': np.std, 
                             'range': lambda x: x.max() - x.min(), 
#                             'skew': 'skew', 
                             'kurtosis': kurtosis, 
                             'alpha': lambda x: (x ** 3).mean(), 
#                             'beta': lambda x: (x ** 4).mean(),
                             })
        if is_save:
 
            if not os.path.exists(os.path.dirname(out_file)):
                os.makedirs(os.path.dirname(out_file))
            df_post.to_pickle(out_file)
            
        return df_post
    
if __name__ == '__main__':
#    in_dir = '/data/20180915'
#    file_list = set([f for f in os.listdir(in_dir) if f.endswith('tdms')])
#    spindle_file_list = set([f for f in file_list if 'spindle' in f])
#    print('all:{} , spindle:{}'.format(len(file_list), len(spindle_file_list)))
##    print(spindle_file_list)
#    file = spindle_file_list.pop()
#    print(file)
#    signalprocess = SignalProcess(data_file='lh_spindle_health_assessment_20180915_004002.tdms', in_dir=in_dir)
#    res = signalprocess.f_envelope_spectrum()
    filename = '/data/20181201_2/M03JS0004/WUDAO/aMIAN/FCFT5/17-20181201095911-log.txt'
    db = mongodb_op.MongodbOp()
#    db.query_dict = {'label' : {'$eq' : 1}, 'youdao' : {'$eq' : False}}
    db.query_dict = {'label' : {'$eq' : 0}}
    res = db.query()
    file_list = set()
    for r in res:
        file = os.path.join(r['a_uri'], r['filename'])
        file_list.add(file)
    dp = preprocess.DataPreprocess(filename=filename)
    df = dp.data_concat()
    signal0 = SignalProcess(df=df)
    post = signal0.t_data_process_by_col()
    post.dropna(axis=0, inplace=True)
    post_30s = post.rolling(20).mean()
    post_1s = post.shift(1)
    post_1s.dropna(axis=0, inplace=True)
    post = post.merge(post_1s, left_index=True, right_index=True, how='inner', suffixes=['', 'last_1_s'])
    post = post.merge(post_30s, left_index=True, right_index=True, how='inner', suffixes=['', 'last_30_s'])
    post.dropna(axis=0, inplace=True)
    res_cut, q_df = dp.data_cut_bins(post, bins=3)
    res_cut
    out_dir = '/data/OUT/QIE/image/f_enve'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig_no = 1937
#    for x in post.index.unique():
#        print('time_{}'.format(x))
#        signal0.df = df.loc[x + pd.Timedelta(5, unit='s')]
#        f, p = signal0.f_envelope_spectrum()
#        plt.figure(fig_no)
#        plt.plot(f, p)
#        plt.savefig(os.path.join(out_dir, '{}.jpg'.format(fig_no)))
#        fig_no += 1
#    for i, f in enumerate(file_list):
#        plt.figure(i)
#        dp._filename = f
#        print(f)
#        data_pre = dp.origin_data().data_split()[0]
#        data_pre = dp.data_drop(data_pre)
#        data_pre = dp.data_drop(data_pre, drop_type='tail', pct=0.2)
#        
#        signal0 = SignalProcess(df=data_pre)
#        signal0.f_eemd()
#        f, p = signal0.f_envelope_spectrum()
#        print('ratio:{}'.format(p[ : f[f <= 750].size].max() / p[f[f > 750].size : ].max()))
#        plt.plot(f, p)
#        signal0 = SignalProcess(df=data_pre['z'])
#        rms = signal0.t_v_rms(r_type='a')
#        print(rms)
#        std = signal0.t_a_std()
#        print(std)
#        mean = signal0.t_a_mean()
#        print(mean)
#        mode = signal0.t_a_mode()
#        print(mode)
#        vrange = signal0.t_a_range()
#        print(vrange)
#        vrange = signal0.f_amp_ratio()
#        print(vrange)
#        rms.plot()
    
    
        


