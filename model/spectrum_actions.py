import numpy as np
import pandas as pd
from numpy import array
from numpy import array, arange, abs as np_abs
from numpy.fft import rfft, rfftfreq
from math import sin, pi
from scipy import signal

def convert_to_counts(sig,value):
    t_shifted = sig - sig.min()
    Counts=0
    for i in  t_shifted:
        if (i<value):
            Counts+=1

    return Counts

def get_spectrum1(time_array,signal_array):
    Fd=np.mean(np.diff(time_array))
    SignalLength=len(signal_array)
    spectr_V = rfft(signal_array)
    freq = rfftfreq(SignalLength,Fd)
    
    V = 2 * np.abs(spectr_V) / SignalLength
    F=freq
    return [F,V]

def get_spectrum2(time_array,signal_array, max_freq=1e6, pad_factor=10):
    
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

def get_spectrum3(t,samples, max_freq=1e6, pad_factor=50, window='hann'):

    Fs = 1.0 / np.mean(np.diff(t))
    N = len(samples)

    # Window
    if window == 'hann':
        w = signal.windows.hann(N)
    elif window == 'hamming':
        w = signal.windows.hamming(N)
    else:
        w = np.ones(N)

    xw = samples * w
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



def refine_peak_frequency(
    x,
    fs,
    iterations=5,
    zoom_points=100,
    window=True
):
    """
    Итеративный поиск частоты максимальной гармоники
    """

    def local_dft(x, fs, freqs):
        """
        x     : сигнал (1D numpy array)
        fs    : частота дискретизации
        freqs : массив частот (Гц), для которых считаем DFT
        """
        n = np.arange(len(x))
        result = np.zeros(len(freqs), dtype=np.complex128)

        for i, f in enumerate(freqs):
            result[i] = np.sum(x * np.exp(-2j * np.pi * f * n / fs))

        return result
    
    N = len(x)

    if window:
        x = x * np.hanning(N)

    # --- 1. Грубый FFT ---
    spectrum = np.fft.rfft(x)
    freqs_fft = np.fft.rfftfreq(N, 1 / fs)

    idx_max = np.argmax(np.abs(spectrum))
    f_center = freqs_fft[idx_max]

    # Начальный шаг по частоте
    df = freqs_fft[1] - freqs_fft[0]

    # --- 2. Итеративный зум ---
    for _ in range(iterations):
        freq_grid = np.linspace(
            f_center - df,
            f_center + df,
            zoom_points
        )

        dft_vals = local_dft(x, fs, freq_grid)
        idx_max = np.argmax(np.abs(dft_vals))

        f_center = freq_grid[idx_max]
        df = (freq_grid[1] - freq_grid[0]) * 2

    return f_center
