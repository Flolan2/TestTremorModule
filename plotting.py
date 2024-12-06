#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotting.py

Contains functions to plot signals and spectrograms.

Created on [Current Date]

@author:
"""

import matplotlib.pyplot as plt
import os
import numpy as np  # Ensure numpy is imported for spectrogram plotting


def plot_signal(time, data, title, xlabel='Time (s)', ylabel='Amplitude', save_path=None):
    """
    Plot a time-series signal.
    
    Parameters:
    - time: 1D array of time points
    - data: 1D array of signal values
    - title: Title of the plot
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    - save_path: Path to save the plot image
    """
    plt.figure(figsize=(10, 4))
    plt.plot(time, data, label='Signal')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Plot saved to {save_path}")
    plt.close()


def plot_spectrogram(frequencies, times, Sxx, title, xlabel='Time (s)', ylabel='Frequency (Hz)', save_path=None):
    """
    Plot a spectrogram.
    
    Parameters:
    - frequencies: Frequencies for spectrogram
    - times: Time segments for spectrogram
    - Sxx: Spectrogram data
    - title: Title of the plot
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    - save_path: Path to save the plot image
    """
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    cbar = plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Spectrogram saved to {save_path}")
    plt.close()
<<<<<<< HEAD
=======

>>>>>>> 09ef9e9 (Sync local directory with GitHub repository)
