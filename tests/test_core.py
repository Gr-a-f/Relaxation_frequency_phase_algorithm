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