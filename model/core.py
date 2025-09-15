import math
import numpy as np
import pandas as pd
from numpy import array
from numpy import array, arange, abs as np_abs
from numpy.fft import rfft, rfftfreq
from math import sin, pi
from scipy import signal

def make_real_data_list(mypath):
    data = pd.read_csv(mypath)
    data.rename(columns = {'in s':'in_s'}, inplace = True)
    if 'C1 in V' in data.columns:
        data.rename(columns = {'C1 in V':'C1_in_V'}, inplace = True)
        ValueArray= array(data.C1_in_V)
    else:
        data.rename(columns = {'C2 in V':'C2_in_V'}, inplace = True)
        ValueArray= array(data.C2_in_V)
    TimeArray =array(data.in_s)
    return TimeArray,ValueArray

def make_microcap_data_list(path):
    df = pd.read_csv(path, skiprows=6, sep=",", engine="python")

    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    time = df.iloc[:, 0].astype(float).values
    values = df.iloc[:, 1].astype(float).values

    return time, values

def get_phase_RC_real(F_main,R,C):
    return math.degrees(np.atan(2*pi*F_main*R*C))

def get_mean_value(sig, range=100):
    central_index=int(len(sig)/2)
    mean_value=np.mean(sig[central_index-range:central_index+range])

    return mean_value

def get_time_of_max_value(time, sig, startpoint=0,endpoint=None):
    if (endpoint==None):
        endpoint=len(sig)

    max_value=np.argmax(sig[startpoint:endpoint])
    time_of_max_value=time[max_value]

    return time_of_max_value

def get_F_rel_mid(sig, F_peak):
    central_index=int(len(sig)/2)
    #phase_central=sig[central_index]
    #f_rel = F_peak * np.cos(phase_central*0.0174533) / np.sin(phase_central*0.0174533)

    phase_mean=np.mean(sig[central_index-100:central_index+100])
    f_rel = F_peak * np.cos(phase_mean*0.0174533) / np.sin(phase_mean*0.0174533)
    return f_rel