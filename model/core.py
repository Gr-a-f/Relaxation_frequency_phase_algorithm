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

def get_phase_RC_real(F_main,R,C):
    return math.degrees(np.atan(2*pi*F_main*R*C))