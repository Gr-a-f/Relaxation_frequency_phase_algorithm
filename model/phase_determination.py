import numpy as np
import pandas as pd
from numpy import array
from numpy import array, arange, abs as np_abs
from numpy.fft import rfft, rfftfreq
from math import sin, pi
from scipy import signal
from model import convert_to_counts


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

def parabolic_peak_polyfit(y, x):
    a, b, c = np.polyfit(x, y, 2)
    if abs(a) < 1e-12:
        return x[1]
    return -b / (2*a)

def get_phase_maxpoint_diff(time, sig1, sig2, f_peak):
    """
    Оценка фазовой разницы по сдвигу максимумов с субсэмпловой интерполяцией.
    Возвращает (t_centers, phase_deg) — списки одинаковой длины.
    """
    T = 1.0 / f_peak
    dt = time[1] - time[0]
    T_counts = max(3, int(round(T / dt)))  # минимум 3 точки на период для интерполяции

    t_result = []
    phase_result = []

    n_periods = len(sig1) // T_counts
    for k in range(n_periods):
        start = k * T_counts
        end = start + T_counts
        if end > len(sig1):
            break

        # максимум sig1
        local1 = sig1[start:end]
        idx1_local = np.argmax(local1)
        idx1 = start + idx1_local
        if idx1 - 1 < 0 or idx1 + 1 >= len(sig1):
            continue
        x1 = time[idx1-1:idx1+2]
        y1 = sig1[idx1-1:idx1+2]
        max_t_sig1 = parabolic_peak_polyfit(y1, x1)

        # максимум sig2 (независимый индекс)
        local2 = sig2[start:end]
        idx2_local = np.argmax(local2)
        idx2 = start + idx2_local
        if idx2 - 1 < 0 or idx2 + 1 >= len(sig2):
            continue
        x2 = time[idx2-1:idx2+2]
        y2 = sig2[idx2-1:idx2+2]
        max_t_sig2 = parabolic_peak_polyfit(y2, x2)

        time_shift = max_t_sig1 - max_t_sig2
        phase_delta = (360.0 * f_peak * time_shift) % 360.0
        phase_delta = abs(phase_delta)
        if phase_delta > 180.0:
            phase_delta = 360.0 - phase_delta

        t_center = 0.5 * (time[start] + time[end-1])
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

def get_phase_xcorr2(time, sig1, sig2, f0, n_periods=10, overlap=0.5):
    """
    Оценка разности фаз между двумя сигналами методом скользящей кросс-корреляции
    с субсэмпловой интерполяцией для высокой точности (до ~0.01 градуса).
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
        peak_index = np.argmax(corr)

        # --- субсэмпловая интерполяция (парабола) ---
        if 0 < peak_index < len(corr) - 1:
            y0, y1, y2 = corr[peak_index - 1], corr[peak_index], corr[peak_index + 1]
            # смещение от целого лага (в отсчётах)
            frac_shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-20)
        else:
            frac_shift = 0.0

        lag_samples = lags[peak_index] + frac_shift

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

def get_phase_xcorr3(
        time,
        sig1,
        sig2,
        f0,
        win_size=None,
        hop_size=None
    ):
    """
    Возвращает фазовый сдвиг в ГРАДУСАХ в диапазоне [-180, 180).
    Остальное — как раньше; если win_size/hop_size = None — выбираются автоматически.
    """

    fs = 1.0 / np.mean(np.diff(time))
    N = len(sig1)

    # автоподбор окна: 5 периодов (в сэмплах), минимум 3, максимум N
    period = fs / f0
    period = max(1, int(round(period)))

    if win_size is None:
        win_size = int(min(N, max(3 * period, 5 * period)))
    if hop_size is None:
        hop_size = max(1, win_size // 2)

    times = []
    phase_array = []
    delay_array = []

    def subdelay(x, y):
        corr = np.correlate(y, x, mode='full')
        lags = np.arange(-len(x) + 1, len(x))
        peak_i = np.argmax(corr)

        if 1 <= peak_i <= len(corr) - 2:
            y1, y2, y3 = corr[peak_i-1:peak_i+2]
            denom = (y1 - 2*y2 + y3)
            delta = 0.5 * (y1 - y3) / denom if denom != 0 else 0.0
        else:
            delta = 0.0

        return (lags[peak_i] + delta) / fs

    for start in range(0, N - win_size + 1, hop_size):
        stop = start + win_size
        x = sig1[start:stop]
        y = sig2[start:stop]

        delay = subdelay(x, y)              # сек
        phase_deg = 360.0 * f0 * delay     # может быть любой величины

        # нормализуем в диапазон [-180, 180)
        phase_signed = ((phase_deg + 180.0) % 360.0) - 180.0

        times.append(time[start + win_size // 2])
        phase_array.append(phase_signed)
        delay_array.append(delay)

    return np.array(times), np.array(phase_array)
