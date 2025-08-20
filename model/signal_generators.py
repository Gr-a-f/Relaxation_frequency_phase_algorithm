import numpy as np
import pandas as pd
from numpy import array
from numpy import array, arange, abs as np_abs
from numpy.fft import rfft, rfftfreq
from math import sin, pi
from scipy import signal


def generate_sin_440():
    fs = 625e6
    duration = 48e-6
    F_main=440e3

    t = np.linspace(0, duration, int(duration * fs))  
    U = np.sin(2 * np.pi * F_main * t)

    return t,U


def generate_meander():
    fs = 1e6
    F_main = 1e3
    duration = 10e-3

    t = np.linspace(0, duration, int(duration * fs))  
    U = signal.square(2 * np.pi * F_main * t)

    return t,U


def RC_transfer(t,U,R,C):
    dU_dt = np.gradient(U, t) 

    I_R = U / R
    I_C = C * dU_dt

    # Общий ток
    I_total = I_R + I_C

    return t, I_total

