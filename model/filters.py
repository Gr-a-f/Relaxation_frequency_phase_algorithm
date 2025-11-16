import numpy as np
import pandas as pd
from numpy import array
from numpy import array, arange, abs as np_abs
from numpy.fft import rfft, rfftfreq
from math import sin, pi
from scipy import signal

def filter_elliptic_bandpass(time, samples, Fcutoff, scope, order=6, rp=0.5, rs=60):
    """
    Полосовой эллиптический фильтр (Cauer filter).

    Parameters
    ----------
    time : ndarray
        Массив времени сигнала.
    samples : ndarray
        Массив значений сигнала.
    Fcutoff : float
        Центральная частота (Гц).
    scope : float
        Полуширина полосы пропускания (Гц).
    order : int, optional
        Порядок фильтра (обычно 2–10).
    rp : float, optional
        Допустимая рябь (ripples) в полосе пропускания, дБ.
    rs : float, optional
        Минимальное подавление вне полосы (dB).

    Returns
    -------
    time_filt : ndarray
        Массив времени (тот же, что на входе).
    samples_filt : ndarray
        Отфильтрованный сигнал.
    """

    # --- Расчёт частот ---
    lowcut = Fcutoff - scope
    highcut = Fcutoff + scope
    Fs = 1 / np.mean(np.diff(time))   # Частота дискретизации
    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq

    # --- Проверка диапазона ---
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid cutoff frequencies: low={low:.4f}, high={high:.4f}")

    # --- Создание фильтра ---
    sos = signal.ellip(order, rp, rs, [low, high], btype='band', output='sos')

    # --- Применяем фильтрацию без сдвига фазы ---
    filtered = signal.sosfiltfilt(sos, samples)

    return time, filtered

def filter_butter_bandpass(time, samples, Fcutoff, scope, order=2):
    lowcut = Fcutoff - scope
    highcut = Fcutoff + scope
    Fs = 1 / np.mean(np.diff(time))
    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq

    if not (0 < low < high < 1):
        raise ValueError(f"Invalid cutoff frequencies: low={low}, high={high}")

    sos = signal.butter(order, [low, high], btype='band', output='sos')
    filtered = signal.sosfiltfilt(sos, samples)
    return time, filtered

def filter_cheby1_bandpass(time, samples, Fcutoff, scope, order=4, rp=1):
    """
    Полосовой фильтр Чебышева I типа.
    
    Parameters
    ----------
    time : ndarray
        Массив времени сигнала.
    samples : ndarray
        Массив значений сигнала.
    Fcutoff : float
        Центральная частота (Гц).
    scope : float
        Полуширина полосы пропускания (Гц).
    order : int, optional
        Порядок фильтра. Обычно 2–8 достаточно.
    rp : float, optional
        Допустимые пульсации (ripples) в полосе пропускания, дБ.
    
    Returns
    -------
    time_filt : ndarray
        Массив времени (тот же, что на входе).
    samples_filt : ndarray
        Отфильтрованный сигнал.
    """
    # Расчёт частот
    lowcut = Fcutoff - scope
    highcut = Fcutoff + scope
    Fs = 1 / np.mean(np.diff(time))     # частота дискретизации
    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq

    # Проверка корректности диапазона
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid cutoff frequencies: low={low}, high={high}")

    # Используем SOS-реализацию (устойчивую численно)
    sos = signal.cheby1(order, rp, [low, high], btype='band', output='sos')

    # Применяем двунаправленную фильтрацию без сдвига фазы
    filtered = signal.sosfiltfilt(sos, samples)

    return time, filtered

def filter_fir(time, samples, Fcutoff, scope, order=None, rp=0.1, rs=60.0,
               trans_width=None, compensate_delay=True, fs=None, zero_phase=False):
    """
    FIR-полосовой фильтр на основе окна Кайзера.

    Параметры
    ----------
    time : np.ndarray
        Вектор времени (в секундах).
    samples : np.ndarray
        Сигнал (вещественный или комплексный).
    Fcutoff : float
        Несущая частота (Гц).
    scope : float
        Полуширина полосы пропускания (Гц), то есть пропуск ±scope от Fcutoff.
    order : int, optional
        Порядок фильтра. Если None — подбирается автоматически через kaiserord.
    rp : float
        Допустимая рябь (ripple) в полосе пропускания, дБ.
    rs : float
        Подавление (attenuation) в полосе подавления, дБ.
    trans_width : float, optional
        Ширина переходной полосы (Гц). Если None — выберется автоматически.
    compensate_delay : bool
        Компенсировать ли задержку фильтра (выравнивать фазу).
    fs : float, optional
        Частота дискретизации. Если None — вычисляется из вектора time.
    zero_phase : bool
        Использовать ли нулевую фазу (двойная фильтрация filtfilt).

    Возвращает
    ----------
    t_filt : np.ndarray
        Время для отфильтрованного сигнала (может быть укорочено, если компенсируется задержка).
    y_filt : np.ndarray
        Отфильтрованный сигнал (комплексный, если вход был комплексный).
    taps : np.ndarray
        Коэффициенты FIR-фильтра.
    """

    # --- Проверка и частота дискретизации ---
    if fs is None:
        dt = np.mean(np.diff(time))
        fs = 1.0 / dt

    nyq = fs / 2.0
    pb_low = Fcutoff - scope
    pb_high = Fcutoff + scope

    if trans_width is None:
        guard = 300_000.0 - scope   # пример резерва до помех (можно скорректировать)
        trans_width = min(50_000.0, guard * 0.5)

    f_pass = [pb_low / nyq, pb_high / nyq]
    f_stop = [(pb_low - trans_width) / nyq, (pb_high + trans_width) / nyq]

    # --- Расчёт параметров фильтра ---
    if order is None:
        tw = trans_width / nyq
        N, beta = signal.kaiserord(rs, tw)
        numtaps = N + 1
    else:
        numtaps = order + 1
        beta = signal.kaiser_beta(rs)

    # --- Проектирование фильтра ---
    taps = signal.firwin(numtaps, [f_pass[0], f_pass[1]],
                         pass_zero=False, window=('kaiser', beta))

    # --- IQ-смещение вниз к 0 Гц ---
    x_shifted = samples * np.exp(-2j * np.pi * Fcutoff * time)

    # --- Фильтрация ---
    if zero_phase:
        y_filt = signal.filtfilt(taps, 1.0, x_shifted)
        delay = 0
    else:
        y_filt = signal.lfilter(taps, 1.0, x_shifted)
        delay = (len(taps) - 1) // 2 if compensate_delay else 0

    # --- Компенсация задержки ---
    if compensate_delay and not zero_phase:
        y_filt = y_filt[delay:]
        t_filt = time[:-delay] if delay < len(time) else time
    else:
        t_filt = time

    return t_filt, y_filt