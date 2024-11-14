# utils.py

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import hilbert, butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def compute_spectrogram(x, fs):
    """Compute the spectrogram of the signal."""
    f, t, Sxx = signal.spectrogram(x, fs)
    return f, Sxx

def compute_power_spectrum(y, fs):
    """Compute the power spectrum of the signal using Welch's method."""
    f, Pxx = signal.welch(y, fs)
    return f, Pxx

def compute_hilbert_amplitude(y):
    """Compute the instantaneous amplitude using Hilbert transform."""
    analytic_signal = hilbert(y)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def meanfilt(x, k):
    """Apply a mean filter to the data."""
    return pd.Series(x).rolling(window=k, center=True, min_periods=1).mean().values

def check_hand_(patient, hands=['left', 'right']):
    """Identify which hands are tracked for the patient based on landmarks."""
    available_hands = []
    landmarks = patient.get('landMarks', {})
    for hand in hands:
        if any(marker.startswith(hand) for marker in landmarks.keys()):
            available_hands.append(hand)
    return available_hands

def identify_active_time_period(structural_features, fs):
    """
    Identify active time periods based on structural features.
    Implement your specific criteria here.
    
    Example Implementation:
    - Identify periods with high variance as active.
    """
    variance = pd.Series(structural_features).rolling(window=int(fs*2), center=True).var().fillna(0)
    active = variance > variance.quantile(0.75)
    active_periods = []
    in_active = False
    start = 0
    for i, val in enumerate(active):
        if val and not in_active:
            in_active = True
            start = i
        elif not val and in_active:
            in_active = False
            end = i
            active_periods.append((start, end))
    return active_periods if active_periods else None
