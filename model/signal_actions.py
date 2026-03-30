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

def generate_sine_random_walk(
    t,
    f0,
    df_pp,      # peak-to-peak нестабильность в Гц
    A=1,
    phase_flat=0
):
    dt = np.mean(np.diff(t))
    N = len(t)

    # 1. белый шум
    w = np.random.randn(N)

    # 2. интеграция → random walk FM
    df = np.cumsum(w)

    # 3. нормировка под нужный диапазон
    df -= np.mean(df)
    df *= df_pp / (np.max(df) - np.min(df))

    phase0 = np.deg2rad(phase_flat)
    # 4. интеграция в фазу
    phase =phase0+ 2 * np.pi * np.cumsum(f0 + df) * dt

    return A * np.sin(phase)

import numpy as np

def generate_common_phase(
    t,
    f0,
    df_pp,
    phase0_deg=0
):
    """
    Общая фаза источника с random walk FM
    """
    dt = np.mean(np.diff(t))
    N = len(t)

    w = np.random.randn(N)
    df = np.cumsum(w)

    # нормировка peak-to-peak
    df -= np.mean(df)
    df *= df_pp / (np.max(df) - np.min(df))

    phase0 = np.deg2rad(phase0_deg)

    phase = phase0 + 2 * np.pi * np.cumsum(f0 + df) * dt
    return phase

def generate_signals_from_phase(
    phase,
    phase_diff_deg,
    A=1.0
):
    """
    phase          — общая фаза (рад)
    phase_diff_deg — разница фаз I относительно U (град)
    """
    dphi = np.deg2rad(phase_diff_deg)

    U = A * np.sin(phase)
    I = A * np.sin(phase + dphi)

    return U, I

def add_SNR(signal, SNR_dB):
    """
    Добавляет белый гауссов шум с заданным SNR (dB)
    """
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(SNR_dB / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    
    return signal + noise

def adc_quantize(
    x,
    n_bits,
    Vref=1.0,
    bipolar=True,
    dither=False
):
    """
    Симуляция идеального АЦП

    x        — входной сигнал (numpy array)
    n_bits   — битность АЦП
    Vref     — опорное напряжение (пик)
    bipolar  — True: [-Vref, +Vref], False: [0, Vref]
    dither   — добавлять ли dither (±0.5 LSB)
    """

    if bipolar:
        xmin, xmax = -Vref, Vref
        levels = 2 ** n_bits
        delta = (xmax - xmin) / levels
    else:
        xmin, xmax = 0.0, Vref
        levels = 2 ** n_bits
        delta = (xmax - xmin) / levels

    # 1. клиппинг
    x_clip = np.clip(x, xmin, xmax - delta)

    # 2. dither (опционально)
    if dither:
        x_clip = x_clip + np.random.uniform(-0.5 * delta, 0.5 * delta, size=len(x))

    # 3. квантование
    q = np.round((x_clip - xmin) / delta)
    x_q = q * delta + xmin

    return x_q


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

