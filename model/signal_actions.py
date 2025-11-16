import math
import numpy as np
import pandas as pd
from numpy import array
from numpy import array, arange, abs as np_abs
from numpy.fft import rfft, rfftfreq
from math import sin, pi
from scipy import signal


def generate_sin(t,F,A=1,phase=0):
    return A * np.sin(2 * np.pi * F * t + np.deg2rad(phase))


def generate_meander():
    fs = 1e6
    F_main = 1e3
    duration = 10e-3

    t = np.linspace(0, duration, int(duration * fs))  
    U = signal.square(2 * np.pi * F_main * t)

    return t,U


def RC_transfer(t,U,R,C):
    dU_dt = np.gradient(U, t, edge_order=2)

    I_R = U / R
    I_C = C * dU_dt

    # Общий ток
    I_total = I_R + I_C

    return t, I_total

def add_realistic_noise(time, signal, F0, Fs,
                        low_freq_amp=0.3,
                        mirror_amp=0.3,
                        harmonic_amp=0.5,
                        white_noise_amp=0.05):
    """
    Добавляет типичные физические помехи к сигналу с несущей F0:
      - низкочастотный дрейф (около 0 Гц)
      - зеркальная гармоника (Fs - F0)
      - вторая гармоника (2*F0)
      - белый шум
    """
    drift = low_freq_amp * np.sin(2*np.pi*1e3*time)          # ~1 кГц
    mirror = mirror_amp * np.sin(2*np.pi*(Fs-F0)*time)
    harmonic = harmonic_amp * np.sin(2*np.pi*(2*F0)*time)
    white = white_noise_amp * np.random.randn(len(signal))
    return signal + drift + mirror + harmonic + white

