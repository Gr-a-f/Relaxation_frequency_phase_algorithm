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

def get_phase_maxpoint_diff(time, sig1, sig2, f_peak):
    """
    Определяет фазовую разницу между двумя сигналами по сдвигу их максимумов
    в каждом периоде основной частоты.
    """

    T = 1 / f_peak
    T_counts = convert_to_counts(time, T)
    
    t_result = []
    phase_result = []

    n_periods = len(sig1) // T_counts
    for T_current in range(1, n_periods):
        time_start = T_counts * (T_current - 1)
        time_end = T_counts * T_current

        max_t_sig1 = time[np.argmax(sig1[time_start:time_end]) + time_start]
        max_t_sig2 = time[np.argmax(sig2[time_start:time_end]) + time_start]

        time_shift = max_t_sig1 - max_t_sig2
        phase_delta = 360 * f_peak * time_shift
        phase_delta = abs(phase_delta) % 360

        if phase_delta > 180:
            phase_delta = 360 - phase_delta

        # Центральная точка окна
        t_center = (time[time_start] + time[time_end]) / 2

        t_result.append(t_center)
        phase_result.append(phase_delta)

    return t_result, phase_result

def get_phase_hilbert(time,sig1,sig2,f_peak=440e3):
    phase1 = np.unwrap(np.angle(signal.hilbert(sig1)))
    phase2 = np.unwrap(np.angle(signal.hilbert(sig2)))
    
    phase_diff = np.rad2deg(phase2 - phase1)
    return time, phase_diff

def get_phase_FFT(time, sig1, sig2, f0, n_periods=10, overlap=0.5):
    """
    Считает разницу фаз между двумя сигналами sig1 и sig2
    через FFT с оконным анализом. Временные точки результата
    ставятся в соответствии с массивом времени time (центр окна).
    
    time : массив времени (такой же длины, как сигналы)
    sig1, sig2 : одномерные массивы сигналов
    fs : частота дискретизации
    f0 : основная частота сигнала (Гц)
    n_periods : сколько периодов сигнала помещать в окно
    overlap : доля перекрытия окон (0.0–0.9)
    """
    n = len(sig1)

    fs = 1.0 / np.mean(np.diff(time))

    # число точек на один период
    T_counts = int(round(fs / f0))
    window_size = T_counts * n_periods

    # если сигнал слишком короткий — ограничим окно
    if window_size > len(sig1):
        window_size = max(1, len(sig1) // 2)
        
    step = int(window_size * (1 - overlap))

    # окно Хэмминга
    win = np.hamming(window_size)

    times = []
    phases = []

    for start in range(0, n - window_size, step):
        end = start + window_size
        win1 = sig1[start:end] * win
        win2 = sig2[start:end] * win

        # FFT
                # FFT
        fft1 = np.fft.fft(win1)
        fft2 = np.fft.fft(win2)
        freqs = np.fft.fftfreq(window_size, 1/fs)

        # индекс ближайшей частоты
        idx = np.argmin(np.abs(freqs - f0))

        # фазовая разница через кросс-спектр
        cross = fft2[idx] * np.conj(fft1[idx])
        diff = np.angle(cross)

        # нормализация [-180, 180]
        diff = np.rad2deg((diff + np.pi) % (2*np.pi) - np.pi)

        # вместо "start/fs" берём центр окна по реальному времени
        t_center = np.mean(time[start:end])

        times.append(t_center)
        phases.append(diff)

    return np.array(times), np.array(phases)

def get_phase_lockin(time, sig1, sig2, f0, n_periods=10):
    """
    Разница фаз между двумя сигналами методом lock-in.

    time : массив времени (той же длины, что и сигналы)
    sig1, sig2 : массивы сигналов одинаковой длины
    fs  : частота дискретизации
    f0  : основная частота
    n_periods : количество периодов в окне усреднения
    """

    fs = 1.0 / np.mean(np.diff(time))

    samples_per_period = int(round(fs / f0))
    window_size = samples_per_period * n_periods

    if window_size > len(sig1):
        window_size = max(1, len(sig1) // 2)

    # опорные сигналы
    ref_cos = np.cos(2 * np.pi * f0 * time)
    ref_sin = np.sin(2 * np.pi * f0 * time)

    # демодуляция для первого сигнала
    I1_raw = sig1 * ref_cos
    Q1_raw = sig1 * ref_sin
    I1 = np.convolve(I1_raw, np.ones(window_size)/window_size, mode="same")
    Q1 = np.convolve(Q1_raw, np.ones(window_size)/window_size, mode="same")
    phase1 = np.unwrap(np.arctan2(Q1, I1))

    # демодуляция для второго сигнала
    I2_raw = sig2 * ref_cos
    Q2_raw = sig2 * ref_sin
    I2 = np.convolve(I2_raw, np.ones(window_size)/window_size, mode="same")
    Q2 = np.convolve(Q2_raw, np.ones(window_size)/window_size, mode="same")
    phase2 = np.unwrap(np.arctan2(Q2, I2))

    # разница фаз
    phase_diff = np.rad2deg(phase1 - phase2)

    return time, phase_diff

def get_phase_xcorr(time, sig1, sig2, f0, n_periods=10, overlap=0.5):
    """
    Оценка разности фаз между двумя сигналами методом скользящей кросс-корреляции.
    Parameters
    ----------
    time : array
        Временной массив, с
    sig1, sig2 : array
        Сигналы одинаковой длины
    f0 : float
        Основная частота, Гц
    n_periods : int, optional
        Длина окна в периодах сигнала (default=10)
    overlap : float [0..1], optional
        Доля перекрытия соседних окон (default=0.5)

    Returns
    -------
    times : array
        Время (центры окон), с
    tau_array : array
        Разность во времени (задержка), с
    phase_array : array
        Разность фаз, градусы (от -180 до 180)
    """
    # шаг дискретизации
    dt = np.mean(np.diff(time))
    fs = 1.0 / dt

    # длина окна в сэмплах
    samples_per_period = int(round(fs / f0))
    window_size = n_periods * samples_per_period
    step_size = int(window_size * (1 - overlap))

    n = len(sig1)
    times = []
    tau_array = []
    phase_array = []

    for start in range(0, n - window_size, step_size):
        end = start + window_size
        x1 = sig1[start:end]
        x2 = sig2[start:end]

        # нормализация
        x1 = (x1 - np.mean(x1)) / (np.std(x1) + 1e-12)
        x2 = (x2 - np.mean(x2)) / (np.std(x2) + 1e-12)

        # кросс-корреляция
        corr = signal.correlate(x1, x2, mode="full")
        lags = np.arange(-len(x1) + 1, len(x1))
        lag_samples = lags[np.argmax(corr)]

        # задержка в секундах
        tau = lag_samples / fs

        # фазовый сдвиг
        phase = (2 * np.pi * f0 * tau) * 180 / np.pi
        phase = (phase + 180) % 360 - 180

        # сохраняем
        times.append((start + end) / 2 * dt)
        tau_array.append(tau)
        phase_array.append(phase)

    return np.array(times), np.array(phase_array)
