import numpy as np
from scipy.signal import butter, lfilter, freqz


def metric_signal_energy(F):
    return np.sum(np.abs(F)**2)/F.size

def metric_signal_smoothness(F):
    return np.std(np.gradient(F))

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
