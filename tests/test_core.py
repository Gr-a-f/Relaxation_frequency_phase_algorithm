import os

from model import *

def test_make_data_list():

    test_dir = os.path.dirname(__file__)
    data_path = os.path.join(test_dir, "data", "RTB2004_CHAN1.csv")

    mydata = make_data_list(data_path)

    assert len(mydata[0]) == len(mydata[1]), "Arrays of points must be of the same size"
    assert type(mydata[0][0]) == np.float64 and type(mydata[1][0]) == np.float64, "Arrays type must be np.float64"


def test_get_spectrum():
    fs = 625e6
    duration = 12e-6
    freq = 440e3

    t = np.arange(0, duration, 1/fs)
    v = np.sin(2 * np.pi * freq * t)

    F, V = get_spectrum3([t, v])

    peak_freq = F[np.argmax(V)]

    assert abs(peak_freq - freq) < 1e3, f"Ожидалось {freq}, получили {peak_freq}"

def test_filter_butter_bandpass():
    fs = 625e6
    duration = 48e-6
    F_main=440e3

    t = np.arange(0, duration, 1/fs)
    v1 = np.sin(2 * np.pi * F_main * t)
    v2= 3*np.sin(2 * np.pi * 100e3 * t)
    v3= 2*np.sin(2 * np.pi * 700e3 * t)

    v_summ=v1+v2+v3

    t_filtered,v_filtered = filter_butter_bandpass([t,v_summ],440e3,100e3)

    F,V=get_spectrum3([t_filtered,v_filtered])

    peak_freq = F[np.argmax(V)]

    assert abs(peak_freq - F_main) < 1e3, f"Ожидалось {F_main}, получили {peak_freq}"
