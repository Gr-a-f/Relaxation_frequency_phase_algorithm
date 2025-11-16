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

def get_time_of_max_value(time, sig, startpoint=0,endpoint=None):
    if (endpoint==None):
        endpoint=len(sig)

    max_value=np.argmax(sig[startpoint:endpoint])
    time_of_max_value=time[max_value]

    return time_of_max_value


def moving_average(time, values, window_size):
    """
    Скользящее среднее с центровкой по времени.
    Возвращает массив времени и значений одинаковой длины.
    """
    if window_size < 1 or window_size > len(values):
        raise ValueError("window_size должен быть >=1 и <= длины массива")

    kernel = np.ones(window_size) / window_size
    smoothed_values = np.convolve(values, kernel, mode='valid')

    # Для каждой точки берем центр окна
    half_window = (window_size - 1) / 2
    new_time = time[np.arange(len(smoothed_values)) + int(np.floor(half_window))]

    return new_time, smoothed_values