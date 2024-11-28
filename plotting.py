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


def plot_signal(time, data, title, xlabel='Time [s]', ylabel='Amplitude', tremor_type='Postural', patient_id=None, save_path=None):
    """
    Plot a time-series signal.
    
    Parameters:
    - time: 1D array of time points
    - data: 1D array of signal values
    - title: Title of the plot
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    - tremor_type: Type of tremor (Postural or Kinematic)
    - patient_id: ID of the patient (optional)
    - save_path: Path to save the plot image

    Returns:
    - None: The function generates a plot and optionally saves it to the specified location.

    """
    base_title = f"{tremor_type.capitalize()} Tremor Signal"
    base_name = f"{tremor_type.lower()}_tremor_signal"

    title = f"{base_title} - {patient_id}" if patient_id else base_title
    filename = f"{base_name}_{patient_id}.svg" if patient_id else f"{base_name}.svg"


    plt.figure(figsize=(10, 4))
    plt.plot(time, data, label='Signal')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Signal plot saved to {save_path}")
    plt.close()


def plot_spectrogram(frequencies, times, Sxx, xlabel='Time [s]', ylabel='Frequency [Hz]', tremor_type='Postural', patient_id=None, save_path=None):
    """
    Plot a spectrogram.
    
    Parameters:
    - frequencies: Frequencies for spectrogram
    - times: Time segments for spectrogram
    - Sxx: Spectrogram data
    - title: Title of the plot
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    - tremor_type: Type of tremor (Postural or Kinematic)
    - patient_id: ID of the patient (optional)
    - save_path: Path to save the plot image

    Returns:
    - None: The function generates a plot and optionally saves it to the specified location.
    """
    base_title = f"{tremor_type.capitalize()} Tremor Spectrogram"
    base_name = f"{tremor_type.lower()}_tremor_spectrogram"

    title = f"{base_title} - {patient_id}" if patient_id else base_title
    filename = f"{base_name}_{patient_id}.svg" if patient_id else f"{base_name}.svg"

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    save_path = os.path.join(save_path, filename)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Power Spectrum saved to {save_path}")
    plt.close()
     
def plot_power_spectrum(frequencies, Pxx, xlabel='Frequency [Hz]', ylabel='Power Spectral Density [V^2/Hz]', tremor_type='Postural',  patient_id=None, save_path=None):
    """
    Plot the power spectrum.

    Parameters:
    - frequencies: Frequencies for power spectrum
    - Pxx: Power spectral density
    - xlabel (str, optional): Label for x-axis
    - ylabel (str, optional): Label for y-axis
    - tremor_type (str, optional): Type of tremor (Postural or Kinematic)
    - patient_id (str, optional): ID of the patient (optional)
    - save_path (str, optional): Path to save the plot image

    Returns:
    - None: The function generates a plot and optionally saves it to the specified location.
    """
    base_title = f"{tremor_type.capitalize()} Tremor Power Spectrum"
    base_name = f"{tremor_type.lower()}_tremor_power_spectrum"

    title = f"{base_title} - {patient_id}" if patient_id else base_title
    filename = f"{base_name}_{patient_id}.svg" if patient_id else f"{base_name}.svg"

    plt.figure(figsize=(10, 4))
    plt.semilogy(frequencies, Pxx)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(save_path, filename)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] Power Spectrum saved to {save_path}")
    plt.close()


