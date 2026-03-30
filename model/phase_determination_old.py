import numpy as np
import pandas as pd
from numpy import array
from numpy import array, arange, abs as np_abs
from numpy.fft import rfft, rfftfreq
from math import sin, pi
from scipy import signal
from model import convert_to_counts
from model import get_spectrum3
from model import get_kde_mode
from scipy.interpolate import interp1d

def get_phase_maxpoint(time, wave1, wave2, F):
    """
    Оценка фазовой разницы между wave1 и wave2
    по сдвигу максимумов с субсэмпловой интерполяцией.

    Возвращает:
        t_centers : ndarray
        phase_deg: ndarray (в диапазоне [-180, 180])
    """

    def parabolic_peak(x, y):
        """Субсэмпловый максимум по 3 точкам"""
        a, b, c = np.polyfit(x, y, 2)
        if abs(a) < 1e-12:
            return x[1]
        t_peak = -b / (2 * a)
        if t_peak < x[0] or t_peak > x[-1]:
            return x[1]
        return t_peak

    dt = time[1] - time[0]
    T = 1.0 / F
    half_win = int(0.5 * T / dt)

    t_centers = []
    phases = []

    for i in range(1, len(wave1) - 1):

        if not (wave1[i-1] < wave1[i] > wave1[i+1]):
            continue

        if i - half_win < 1 or i + half_win >= len(wave1) - 1:
            continue

        t1 = parabolic_peak(
            time[i-1:i+2],
            wave1[i-1:i+2]
        )

        j0 = i - half_win
        j1 = i + half_win

        local2 = wave2[j0:j1]
        idx2 = np.argmax(local2) + j0

        if idx2 - 1 < 0 or idx2 + 1 >= len(wave2):
            continue

        t2 = parabolic_peak(
            time[idx2-1:idx2+2],
            wave2[idx2-1:idx2+2]
        )

        dt_phase = t2 - t1
        phase = 360.0 * F * dt_phase
        phase = (phase + 180) % 360 - 180

        t_centers.append(t1)
        phases.append(phase)

    KDE=get_kde_mode(phases)
    return KDE
    #return np.array(t_centers), np.array(phases) #если нужно вернуть массив времени и фаз на каждый период


def get_phase_zero_crossings(
        time,
        sig1,
        sig2,
        f0,
        win_size=None,
        hop_size=None
    ):
    """
    Фаза между двумя сигналами по zero-crossings
    """
    def zero_crossings_interp(time, signal):
        """
        Возвращает массив времен zero-crossing (от - к +)
        с линейной интерполяцией
        """
        zc_times = []

        for i in range(len(signal) - 1):
            if signal[i] < 0 and signal[i + 1] >= 0:
                # линейная интерполяция
                t1, t2 = time[i], time[i + 1]
                y1, y2 = signal[i], signal[i + 1]

                t_zero = t1 - y1 * (t2 - t1) / (y2 - y1)
                zc_times.append(t_zero)

        return np.array(zc_times)
    
    zc1 = zero_crossings_interp(time, sig1)
    zc2 = zero_crossings_interp(time, sig2)

    if len(zc1) == 0 or len(zc2) == 0:
        return time, np.array([])

    # сопоставляем zero-crossings по времени
    phases = []
    times = []

    j = 0
    for t1 in zc1:
        # ищем ближайший zero-crossing второго сигнала
        while j + 1 < len(zc2) and abs(zc2[j + 1] - t1) < abs(zc2[j] - t1):
            j += 1

        dt = zc2[j] - t1
        phi = 360.0 * f0 * dt

        phases.append(phi)
        times.append(t1)

    KDE=get_kde_mode(phases)
    return KDE
    #return np.array(times), np.array(phases) #если нужно вернуть массив времени и фаз на каждый период

def get_phase_hilbert(time, sig1, sig2, f_peak=440e3, n_periods=10):
    """
    Оценка фазовой разницы методом Гилберта.
    """

    phase1 = np.unwrap(np.angle(signal.hilbert(sig1)))
    phase2 = np.unwrap(np.angle(signal.hilbert(sig2)))
    
    phases = np.rad2deg(phase2 - phase1)
    phases = np.rad2deg(np.unwrap(np.angle(signal.hilbert(sig2)) - np.angle(signal.hilbert(sig1))))
    phases = (phases + 180) % 360 - 180

    KDE=get_kde_mode(phases)
    return KDE
    #return time, phases #если нужно вернуть массив времени и фаз на каждый период

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

def get_phase_xcorr1(time, sig1, sig2, f0, n_periods=10, overlap=0.5):
    """
    Оценка разности фаз между двумя сигналами методом скользящей кросс-корреляции.
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
    с субсэмпловой интерполяцией для высокой точности.
    """
    # шаг дискретизации
    dt = np.mean(np.diff(time))
    fs = 1.0 / dt

    # длина окна в сэмплах
    samples_per_period = int(round(fs / f0))
    window_size = n_periods * samples_per_period

    # если сигнал слишком короткий — ограничим окно
    if window_size > len(sig1):
        window_size = max(1, len(sig1) // 2)

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

def get_phase_xcor_zoom(
    time,
    sig1,
    sig2,
    f0,
    n_grid=10,
    n_iter=6,
    tau_range=None,
):
    """
    Итеративный поиск временного сдвига
    с зумированием по максимуму корреляции.

    Parameters
    ----------
    n_grid : int
        Число точек на каждой итерации
    n_iter : int
        Число итераций зума
    """

    T = 1.0 / f0

    if tau_range is None:
        tau_range = (-T/2, T/2)

    interp_sig2 = interp1d(
        time,
        sig2,
        kind="cubic",
        bounds_error=False,
        fill_value=0.0
    )

    x1n = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-12)

    tau_min, tau_max = tau_range

    for _ in range(n_iter):

        tau_grid = np.linspace(tau_min, tau_max, n_grid)
        corr = np.zeros(n_grid)

        for i, tau in enumerate(tau_grid):
            x2s = interp_sig2(time + tau)
            x2n = (x2s - np.mean(x2s)) / (np.std(x2s) + 1e-12)

            corr[i] =np.mean(x1n * x2n)

        k = np.argmax(corr)

        # защита от краёв
        if k == 0:
            tau_min, tau_max = tau_grid[0], tau_grid[1]
        elif k == n_grid - 1:
            tau_min, tau_max = tau_grid[-2], tau_grid[-1]
        else:
            tau_min, tau_max = tau_grid[k - 1], tau_grid[k + 1]

    tau_best = 0.5 * (tau_min + tau_max)
    phase_deg = 360 * f0 * tau_best
    return phase_deg


def get_phase_VVV(
        time,
        sig1,
        sig2,
        f0=None,
        win_size=None,
        hop_size=None
    ):
    def get_phase_solo(t, wave):
        F, V = get_spectrum3(t, wave)
        idx = np.argmax(V)
        F_peak = F[idx]

        ref_sin = np.sin(2*np.pi*f0*time)
        ref_cos = np.cos(2*np.pi*f0*time)

        I = np.mean(wave * ref_cos)
        Q = np.mean(wave * ref_sin)

        phi = np.rad2deg(np.arctan2(Q, I))
        return phi
        
    phase1=get_phase_solo(time,sig1)
    phase2=get_phase_solo(time,sig2)
    dela_phase = np.array([phase1 - phase2])

    return time,dela_phase

import numpy as np

def get_phase_VNA(
        time,
        sig1,
        sig2,
        f0,
        win_size=None,
        hop_size=None
    ):

    # опорный комплексный сигнал
    ref = np.exp(-1j * 2 * np.pi * f0 * time)

    # комплексная демодуляция
    Z1 = sig1 * ref
    Z2 = sig2 * ref

    # усреднение (если окно не задано — по всему сигналу)
    if win_size is None:
        C1 = np.mean(Z1)
        C2 = np.mean(Z2)
        phase = np.angle(C1 / C2)
        return time, np.atleast_1d(np.rad2deg(phase))

    # оконный VNA (для массива фаз)
    if hop_size is None:
        hop_size = win_size // 2

    phases = []
    times = []

    for start in range(0, len(time) - win_size, hop_size):
        stop = start + win_size

        C1 = np.mean(Z1[start:stop])
        C2 = np.mean(Z2[start:stop])

        phase = np.angle(C1 / C2)
        phases.append(np.rad2deg(phase))
        times.append(np.mean(time[start:stop]))

    return np.array(times), np.array(phases)


def get_phase_PPV(
        time,
        sig1,
        sig2,
        f0=None,
        win_size=None,
        hop_size=None
    ):
    def get_phase_solo(t, wave):
        ref_sin = np.sin(2*np.pi*f0*time)
        ref_cos = np.cos(2*np.pi*f0*time)

        I = np.mean(wave * ref_cos)
        Q = np.mean(wave * ref_sin)

        phi = np.rad2deg(np.arctan2(Q, I))
        return phi
        
    phase1=get_phase_solo(time,sig1)
    phase2=get_phase_solo(time,sig2)
    dela_phase = np.array([phase1 - phase2])

    return time,dela_phase

def get_phase_swff(time, sig, f0):
    w = 2*np.pi*f0
    s = np.sin(w*time)
    c = np.cos(w*time)

    B = np.mean(sig * s)
    C = np.mean(sig * c)

    return np.degrees(np.arctan2(C, B))