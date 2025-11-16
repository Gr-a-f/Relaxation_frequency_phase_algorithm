import numpy as np
import pandas as pd
from numpy import array
from numpy import array, arange, abs as np_abs
from numpy.fft import rfft, rfftfreq
from math import sin, pi
from scipy import signal
from scipy.stats import gaussian_kde
import math


def get_phase_RC_real(F_main,R,C):
    return math.degrees(np.atan(2*pi*F_main*R*C))

def get_mean_value(sig,point=None, range=100):
    if (point==None):
        point = len(sig)/2

    central_index=int(len(sig)/2)
    mean_value=np.mean(sig[central_index-range:central_index+range])

    return mean_value


def get_F_rel(phase, F_peak):
    f_rel = F_peak * np.cos(phase*0.0174533) / np.sin(phase*0.0174533)
    
    return f_rel

def get_F_rel_mid(sig, F_peak):
    central_index=int(len(sig)/2)
    #phase_central=sig[central_index]
    #f_rel = F_peak * np.cos(phase_central*0.0174533) / np.sin(phase_central*0.0174533)

    phase_mean=np.mean(sig[central_index-100:central_index+100])
    f_rel = F_peak * np.cos(phase_mean*0.0174533) / np.sin(phase_mean*0.0174533)
    return f_rel

def get_kde_mode(data, bandwidth=None):
    kde = gaussian_kde(data, bw_method=bandwidth)
    xs = np.linspace(min(data), max(data), 1000)
    ys = kde(xs)
    return xs[np.argmax(ys)]

def print_full_stats(F_main,*arrays):
    for i, arr in enumerate(arrays, start=1):
        phase_mean = np.mean(arr)
        phase_mode = get_kde_mode(arr)

        F_rel_mean=get_F_rel(phase_mean,F_main)
        F_rel_mode=get_F_rel(phase_mode,F_main)

        print(f"Array {i}: Mean phase = {phase_mean:.4f}, KDE mode phase = {phase_mode:.4f}")
        print(f"Array {i}: Mean Frel = {F_rel_mean:.4f}, KDE mode Frel = {F_rel_mode:.4f}")


def get_mean_mode(*arrays):
    for i, arr in enumerate(arrays, start=1):
        all_mean=[]
        all_mode=[]

        all_mean.append(np.mean(arr))
        all_mode.append(get_kde_mode(arr))
    
    middle_mean=np.mean(all_mean)
    middle_mode=np.mean(all_mode)

    return middle_mean,middle_mode

def get_mean_from_all(*arrays):
    for i, arr in enumerate(arrays, start=1):
        all_mean=[]

        all_mean.append(np.mean(arr))
    
    middle_mean=np.mean(all_mean)

    return middle_mean
    

    

        

    