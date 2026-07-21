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

def get_phase_MP(time, wave1, wave2, F): #Maxpoint
    """
    Оценка фазовой разницы между wave1 и wave2
    по первой паре максимумов.
    """

    def parabolic_peak(x, y):
        if len(x) < 3:
            return x[len(x)//2]
        a, b, c = np.polyfit(x, y, 2)
        if abs(a) < 1e-12:
            return x[1]
        t_peak = -b / (2 * a)
        return np.clip(t_peak, x[0], x[-1])

    dt = time[1] - time[0]
    T = 1.0 / F
    half_win = max(1, int(0.5 * T / dt))  # ⬅️ КРИТИЧНО

    # ---------- 1. Попытка: строгие локальные максимумы ----------
    for i in range(1, len(wave1) - 1):

        if not (wave1[i-1] <= wave1[i] >= wave1[i+1]):  # ⬅️ <= вместо <
            continue

        if i - half_win < 1 or i + half_win >= len(wave1) - 1:
            continue

        t1 = parabolic_peak(time[i-1:i+2], wave1[i-1:i+2])

        j0 = max(0, i - half_win)
        j1 = min(len(wave2), i + half_win)

        local2 = wave2[j0:j1]
        idx2 = np.argmax(local2) + j0

        t2 = parabolic_peak(time[idx2-1:idx2+2], wave2[idx2-1:idx2+2])

        phase = 360.0 * F * (t2 - t1)
        return (phase + 180) % 360 - 180

    # ---------- 2. ФОЛБЭК: первый период ----------
    n_period = int(T / dt)
    n_period = max(3, min(n_period, len(wave1)//2))

    idx1 = np.argmax(wave1[:n_period])
    idx2 = np.argmax(wave2[:n_period])

    t1 = parabolic_peak(time[idx1-1:idx1+2], wave1[idx1-1:idx1+2])
    t2 = parabolic_peak(time[idx2-1:idx2+2], wave2[idx2-1:idx2+2])

    phase = 360.0 * F * (t2 - t1)
    return (phase + 180) % 360 - 180



def get_phase_ZCRR(
        time,
        sig1,
        sig2,
        f0,
        win_size=None,
        hop_size=None
    ): #Zero_crossings
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

def get_phase_HB(time, sig1, sig2, f_peak=440e3, n_periods=10): #Hilbert
    """
    Оценка фазовой разницы методом Гилберта.
    """

    phase1 = np.unwrap(np.angle(signal.hilbert(sig1)))
    phase2 = np.unwrap(np.angle(signal.hilbert(sig2)))
    
    phases = np.rad2deg(phase2 - phase1)
    phases = np.rad2deg(np.unwrap(np.angle(signal.hilbert(sig1)) - np.angle(signal.hilbert(sig2))))
    phases = (phases + 180) % 360 - 180

    KDE=get_kde_mode(phases)
    return KDE
    #return time, phases #если нужно вернуть массив времени и фаз на каждый период

def get_phase_FFT(time, sig1, sig2, f0):

    # окно
    window = np.hanning(len(sig1))
    sig1 = sig1 * window
    sig2 = sig2 * window

    # точный DFT на частоте f0
    exp = np.exp(-1j * 2 * np.pi * f0 * time)

    X1 = np.sum(sig1 * exp)
    X2 = np.sum(sig2 * exp)

    cross = X1 * np.conj(X2)
    phase = np.angle(cross)

    return np.rad2deg(phase)

def get_phase_XCOR(
    time,
    sig1,
    sig2,
    f0,
    n_grid=10,
    n_iter=10,
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

def get_phase_LI(time, sig1, sig2, f0, n_periods=10): #Lockin
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

    # демодуляция для второго сигнала
    I2_raw = sig2 * ref_cos
    Q2_raw = sig2 * ref_sin
    I2 = np.convolve(I2_raw, np.ones(window_size)/window_size, mode="same")
    Q2 = np.convolve(Q2_raw, np.ones(window_size)/window_size, mode="same")
    
    C1 = I1 + 1j * Q1
    C2 = I2 + 1j * Q2
    phase_diff = np.rad2deg(np.angle(C2 * np.conj(C1)))
    
    return get_kde_mode(phase_diff)
    #return time, phase_diff

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
    dela_phase = phase2 - phase1

    return dela_phase

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

    return np.array(phases)

def wrap_phase_deg(phi):
    """
    Приведение фазы к диапазону (-180, 180]
    """
    return (phi + 180) % 360 - 180

def get_phase_swff(time, sig1, sig2, f0, win_size=None, hop_size=None):

    def get_phase_local(time, sig, f0):
        w = 2*np.pi*f0
        s = np.sin(w*time)
        c = np.cos(w*time)

        B = np.mean(sig * s)
        C = np.mean(sig * c)

        return np.degrees(np.arctan2(C, B))

    phase1 = get_phase_local(time, sig1, f0)
    phase2 = get_phase_local(time, sig2, f0)

    phase_diff = phase1 - phase2
    return wrap_phase_deg(phase_diff)



def get_phase_swf3p(time, sig1, sig2, f0, win_size=None, hop_size=None):

    def swf3p_phase(time, sig, f0):
        w = 2 * np.pi * f0

        S = np.sin(w * time)
        C = np.cos(w * time)
        O = np.ones_like(time)

        X = np.column_stack((S, C, O))
        a, b, offset = np.linalg.lstsq(X, sig, rcond=None)[0]

        return np.rad2deg(np.arctan2(b, a))

    phase1 = swf3p_phase(time, sig1, f0)
    phase2 = swf3p_phase(time, sig2, f0)

    phase_diff = phase1 - phase2
    return wrap_phase_deg(phase_diff)


def get_phase_swff4p(time, sig1, sig2, f0, win_size=None, hop_size=None):

    def swff4p_phase(time, sig, f0): 
        w = 2 * np.pi * f0

        S = np.sin(w * time)
        C = np.cos(w * time)
        O = np.ones_like(time)
        T = time - np.mean(time)

        X = np.column_stack((S, C, O, T))
        a, b, offset, trend = np.linalg.lstsq(X, sig, rcond=None)[0]

        return np.rad2deg(np.arctan2(b, a))

    phase1 = swff4p_phase(time, sig1, f0)
    phase2 = swff4p_phase(time, sig2, f0)

    phase_diff = phase1 - phase2
    return wrap_phase_deg(phase_diff)


def get_phase_SWFR(time, sig1, sig2, f0, win_size=None, hop_size=None):

    def swfr_phase(time, sig, f0):
        w = 2 * np.pi * f0

        S = np.sin(w * time)
        C = np.cos(w * time)
        O = np.ones_like(time)

        X = np.column_stack((S, C, O))
        a, b, offset = np.linalg.lstsq(X, sig, rcond=None)[0]

        phase0 = np.arctan2(b, a)
        A = np.hypot(a, b)

        sig_fit = a * S + b * C + offset
        r = sig - sig_fit

        dphi = A * np.cos(w * time + phase0)
        delta_phi = np.dot(r, dphi) / np.dot(dphi, dphi)

        return np.rad2deg(phase0 + delta_phi)

    phase1 = swfr_phase(time, sig1, f0)
    phase2 = swfr_phase(time, sig2, f0)

    phase_diff = phase1 - phase2
    return wrap_phase_deg(phase_diff)

