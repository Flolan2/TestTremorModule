# feature_extraction.py

import pandas as pd
from tqdm import tqdm
from tremor_extraction import extract_tremor_features_from_signal, extract_proximal_tremor_features

def extract_features(pc, tremor_type='postural', plot=False, save_plots=False):
    """
    Extract tremor features for all patients.
    
    Parameters:
    - pc: Patient collection (dictionary loaded from JSON)
    - tremor_type: Type of tremor to analyze ('postural' or 'kinematic')
    - plot: Boolean to indicate if plotting is desired
    - save_plots: Boolean to indicate if plots should be saved
    
    Returns:
    - Pandas DataFrame containing features for all patients
    """
    features_df = pd.DataFrame(index=[p['patient_id'] for p in pc['patients']])
    
    print(f'Extracting {tremor_type} tremor features ... ')
    for p in tqdm(pc['patients'], total=len(pc['patients'])):
        patient_id = p['patient_id']
        fs = p.get('sampling_frequency', 25)  # Default to 25 Hz
        
        # Select data based on tremor type
        if tremor_type == 'kinematic':
            data = p.get('linePlot', {}).get('data', [])
        elif tremor_type == 'postural':
            data = p.get('rawData', {}).get('data', [])
        else:
            raise ValueError("Invalid tremor_type. Choose 'postural' or 'kinematic'.")
        
        if not data:
            print(f"No data for patient {patient_id} in {tremor_type} tremor. Skipping...")
            continue
        
        data = pd.Series(data).interpolate().dropna().values  # Handle missing data
        
        # Extract features
        features = extract_tremor_features_from_signal(
            data=data,
            fs=fs,
            tremor_type=tremor_type,
            plot=plot,
            save_plots=save_plots,
            patient_id=patient_id
        )
        
        # Assign to DataFrame
        for key, value in features.items():
            column_name = f"{tremor_type}_{key}"
            features_df.loc[patient_id, column_name] = value
    
    return features_df
