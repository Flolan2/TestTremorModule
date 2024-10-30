# preprocessing.py

import numpy as np
from skimage.restoration import denoise_tv_chambolle
from scipy.signal import detrend
from utils import meanfilt, butter_bandpass_filter

def preprocess_signal(data, fs, lowcut=3, highcut=12, order=5, k=21, remove_tv=False):
    """
    Preprocess the signal by detrending, smoothing, and filtering.

    Parameters:
    - data: 1D numpy array of the signal
    - fs: Sampling frequency
    - lowcut: Low frequency cutoff for bandpass filter
    - highcut: High frequency cutoff for bandpass filter
    - order: Order of the Butterworth filter
    - k: Window size for mean filter
    - remove_tv: Boolean to indicate if total variation smoothing should be applied

    Returns:
    - filtered: Preprocessed signal
    """
    # Detrend the data
    detrended = detrend(data)

    # Smooth the data using mean filter
    smoothed = meanfilt(detrended, k)
    detrended_smoothed = detrended - smoothed
    detrended_smoothed = detrend(detrended_smoothed)

    # Optionally remove total variation using skimage
    if remove_tv:
        # weight parameter controls the degree of denoising
        smoothed_tv = denoise_tv_chambolle(detrended_smoothed, weight=0.1)
        detrended_smoothed -= smoothed_tv

    # Apply bandpass filter
    filtered = butter_bandpass_filter(detrended_smoothed, lowcut, highcut, fs, order=order)

    return filtered
