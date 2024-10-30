#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:02:56 2024

@author: Lange_L
"""

# plotting.py

import matplotlib.pyplot as plt
import os

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
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_spectrogram(frequencies, times, Sxx, title, xlabel='Frequency (Hz)', ylabel='Time (s)', save_path=None):
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
    plt.colorbar(label='Intensity [dB]')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
