# tremor_extraction.py

from utils import compute_power_spectrum, compute_spectrogram, compute_hilbert_amplitude
from preprocessing import preprocess_signal

def extract_tremor_features_from_signal(data, fs, tremor_type='postural', plot=False, save_plots=False, patient_id=None):
    """
    Extract tremor features from a single signal without requiring patient_id.
    
    Parameters:
    - data: 1D numpy array of the signal
    - fs: Sampling frequency
    - tremor_type: Type of tremor ('postural' or 'kinematic')
    - plot: Boolean to indicate if plotting is desired
    - save_plots: Boolean to indicate if plots should be saved
    - patient_id: Identifier for the patient (used in plot titles), optional
    
    Returns:
    - Dictionary of extracted features
    """
    # Preprocess the signal
    filtered = preprocess_signal(data, fs, remove_tv=(tremor_type == 'postural'))
    
    # Compute power spectrum
    f_x, P_x = compute_power_spectrum(filtered, fs)
    
    # Compute spectrogram
    f_spectrogram_x, Sxx_x = compute_spectrogram(filtered, fs)
    
    # Compute Hilbert transform for amplitude envelope
    amplitude_envelope_x = compute_hilbert_amplitude(filtered)
    
    features = {}
    
    # Hilbert features
    features['hilbert_max_amplitude'] = 2 * amplitude_envelope_x.max()
    features['hilbert_mean_amplitude'] = 2 * amplitude_envelope_x.mean()
    
    # Spectrogram features
    mean_freq_x = f_spectrogram_x[Sxx_x.mean(axis=0).argmax()] 
    max_freq_x = f_spectrogram_x[Sxx_x.max(axis=0).argmax()]
    max_amplitude_x = Sxx_x.max()
    features['spectrogram_mean_frequency'] = mean_freq_x
    features['spectrogram_max_frequency'] = max_freq_x
    features['spectrogram_max_amplitude'] = 2 * max_amplitude_x
    
    # Power spectrum features
    dom_f_idx = P_x.argmax()
    dominant_frequency = f_x[dom_f_idx]
    amplitude_x = P_x[dom_f_idx]
    features['power_spectral_dominant_frequency'] = dominant_frequency
    features['power_spectral_max_amplitude'] = 2 * amplitude_x
    
    # Plotting
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(filtered, label='Filtered Signal')
        
        # Adjust plot title
        if patient_id is not None:
            plt.title(f"{tremor_type.capitalize()} Tremor - {patient_id}")
        else:
            plt.title(f"{tremor_type.capitalize()} Tremor")
        
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()
        
        if save_plots:
            import os
            os.makedirs('./plots', exist_ok=True)
            # Adjust filename to avoid issues when patient_id is None
            if patient_id is not None:
                filename = f'./plots/{tremor_type}_tremor_{patient_id}.svg'
            else:
                filename = f'./plots/{tremor_type}_tremor.svg'
            plt.savefig(filename)
        plt.close()
    
    return features


def extract_proximal_tremor_features(pc, plot=False, save_plots=False):
    """
    Extract proximal tremor features from the patient collection without requiring patient_id.
    
    Parameters:
    - pc: Patient collection data (should contain 'patients' key)
    - plot: Boolean to indicate if plotting is desired
    - save_plots: Boolean to indicate if plots should be saved
    
    Returns:
    - Pandas DataFrame containing proximal tremor features
    """
    import pandas as pd
    from tqdm import tqdm
    import numpy as np
    
    # Check if 'patients' key exists
    if 'patients' not in pc:
        print("No 'patients' data found in pc.")
        return pd.DataFrame()
    
    # Initialize an empty DataFrame
    features_df = pd.DataFrame()
    
    print('Extracting proximal tremor features ... ')
    for idx, p in tqdm(enumerate(pc['patients']), total=len(pc['patients'])):
        # Use index as a unique identifier if patient_id is not available
        patient_identifier = p.get('patient_id', f'Patient_{idx}')
        
        peaks = p.get('peaks', {})
        valleys = p.get('valleys', {})
        time_peaks = peaks.get('time', [])
        time_valleys = valleys.get('time', [])
        data_peaks = peaks.get('data', [])
        data_valleys = valleys.get('data', [])
        
        # Example Features:
        num_peaks = len(data_peaks)
        num_valleys = len(data_valleys)
        features_df.loc[patient_identifier, 'num_peaks'] = num_peaks
        features_df.loc[patient_identifier, 'num_valleys'] = num_valleys
    
        # Additional Features: Amplitude of peaks and valleys
        if num_peaks > 0:
            max_peak = max(data_peaks)
            min_peak = min(data_peaks)
            features_df.loc[patient_identifier, 'max_peak_amplitude'] = max_peak
            features_df.loc[patient_identifier, 'min_peak_amplitude'] = min_peak
        else:
            features_df.loc[patient_identifier, 'max_peak_amplitude'] = np.nan
            features_df.loc[patient_identifier, 'min_peak_amplitude'] = np.nan
    
        if num_valleys > 0:
            max_valley = max(data_valleys)
            min_valley = min(data_valleys)
            features_df.loc[patient_identifier, 'max_valley_amplitude'] = max_valley
            features_df.loc[patient_identifier, 'min_valley_amplitude'] = min_valley
        else:
            features_df.loc[patient_identifier, 'max_valley_amplitude'] = np.nan
            features_df.loc[patient_identifier, 'min_valley_amplitude'] = np.nan
    
    return features_df
