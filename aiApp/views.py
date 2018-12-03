import os
import numpy as np
from django.shortcuts import render
from .tools import tdms_op, signalProcess, txt_process
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from pyecharts import Line3D, Line

# Create your views here.

REMOTE_HOST = "/home/yzp1011/code/competion/static/pyecharts/assets/js"

def index(request):
    template = loader.get_template('t.html')
    l0 = specturm('vibration_1')
    context = dict(
        myechart=l0.render_embed(),
        script_list=l0.get_js_dependencies()
    )
    return HttpResponse(template.render(context, request))

def specturm(col_name, channel_num=1, part='1'):
    in_dir = '/data/20180915'
#    file_list = set([f for f in os.listdir(in_dir) if f.endswith('tdms')])
#    spindle_file_list = set([f for f in file_list if 'spindle' in f])
#    file = spindle_file_list.pop()
#    signalprocess = signalProcess.SignalProcess(data_file='lh_spindle_health_assessment_20180915_004002.tdms', in_dir=in_dir)
#    f, p = signalprocess.f_envelope_spectrum()
    tp = txt_process.TxtProcess()
    tp.filename = '/data/20181201_1/M03JS0004/WUDAO/aMIAN/FCFT5/17-20181201095911-log.txt'
    df = tp.read_txt()
#    df = df.sample(frac=0.1)
    x_val = np.array(df.index)
    y_val = np.array(df.z)
#    signal = signalProcess.SignalProcess(channel_num=1, part='1')
#    res = signal.fft_df()
    line = Line('频谱图')
    line.add(col_name, x_val, y_val, xaxis_type='value', is_datazoom_show=True, is_datazoom_extra_show=True)
#    print(f)
#    print(p)
#    line.add(col_name, f, p, is_datazoom_show=True, is_datazoom_extra_show=True)
    return line