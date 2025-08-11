import numpy as np
import pandas as pd
from numpy import array
from numpy import array, arange, abs as np_abs
from numpy.fft import rfft, rfftfreq
from math import sin, pi
from scipy import signal

def get_spectrum1(MyList):
    time_array = MyList[0]
    signal_array = MyList[1]
    Fd=np.mean(np.diff(time_array))
    SignalLength=len(MyList[0])
    spectr_V = rfft(signal_array)
    freq = rfftfreq(SignalLength,Fd)
    
    V = 2 * np.abs(spectr_V) / SignalLength
    F=freq
    return [F,V]

def get_spectrum2(my_list, max_freq=1e6, pad_factor=10):
    time_array = my_list[0]
    signal_array = my_list[1]
    
    # Частота дискретизации
    Fs = 1 / np.mean(np.diff(time_array))
    
    # Zero-padding
    N = len(signal_array)
    signal_padded =np.pad(signal_array, (0, pad_factor * N), 'constant')
    
    # FFT
    freq = rfftfreq(len(signal_padded), d=1/Fs)
    spectrum = np.abs(rfft(signal_padded)) * 2 / N  # нормировка на N исходного
    
    # Ограничение диапазона
    mask = freq <= max_freq
    return freq[mask], spectrum[mask]

def get_spectrum3(my_list, max_freq=1e6, pad_factor=50, window='hann'):
    t = np.asarray(my_list[0])
    x = np.asarray(my_list[1])

    Fs = 1.0 / np.mean(np.diff(t))
    N = len(x)

    # Window
    if window == 'hann':
        w = signal.windows.hann(N)
    elif window == 'hamming':
        w = signal.windows.hamming(N)
    else:
        w = np.ones(N)

    xw = x * w
    # Нормировка окна чтобы амплитуды были сопоставимы с исходным сигналом
    correction = 1.0 / (w.mean())

    # Zero-padding
    xpad = np.pad(xw, (0, pad_factor * N), 'constant')
    Npad = len(xpad)

    F = rfftfreq(Npad, d=1/Fs)
    S = np.abs(rfft(xpad)) * 2.0 / N  # нормировка на исходную длину

    # применяем коррекцию окна
    S *= correction

    mask = F <= max_freq
    return F[mask], S[mask]

def parabolic_peak(freqs, spectrum, peak_idx):
    # параболическая интерполяция по трём точкам: idx-1, idx, idx+1
    k = peak_idx
    if k <= 0 or k >= len(spectrum)-1:
        return freqs[k]
    y0, y1, y2 = spectrum[k-1], spectrum[k], spectrum[k+1]
    # смещение вершины параболы относительно центра
    denom = (y0 - 2*y1 + y2)
    if denom == 0:
        return freqs[k]
    dx = 0.5 * (y0 - y2) / denom
    df = freqs[1] - freqs[0]
    return freqs[k] + dx * df