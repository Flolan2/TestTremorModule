import pandas as pd
from tremor_extraction import extract_tremor_features_from_signal

def extract_features(pc, tremor_type='postural', plot=False, save_plots=False):
    """
    Extract tremor features from a single patient's JSON file without requiring patient_id.
    
    Parameters:
    - pc: Data dictionary loaded from JSON (contains global keys like 'linePlot', 'rawData')
    - tremor_type: Type of tremor to analyze ('postural' or 'kinematic')
    - plot: Boolean to indicate if plotting is desired
    - save_plots: Boolean to indicate if plots should be saved
    
    Returns:
    - Pandas DataFrame containing features
    """
    features_df = pd.DataFrame()
    
    print(f'Extracting {tremor_type} tremor features...')
    
    # Select data based on tremor_type
    if tremor_type == 'kinematic':
        data = pc.get('linePlot', {}).get('data', [])
    elif tremor_type == 'postural':
        data = pc.get('rawData', {}).get('data', [])
    else:
        raise ValueError("Invalid tremor_type. Choose 'postural' or 'kinematic'.")
    
    if not data:
        print(f"No data available in {tremor_type} tremor type. Skipping...")
        return features_df
    
    # Interpolate and remove missing values
    data = pd.Series(data).interpolate().dropna().values

    # Extract features
    features = extract_tremor_features_from_signal(
        data=data,
        fs=25,  # Standard sampling frequency
        tremor_type=tremor_type,
        plot=plot,
        save_plots=save_plots
        # Removed patient_id parameter
    )
    
    # Add features to the DataFrame
    for key, value in features.items():
        column_name = f"{tremor_type}_{key}"
        features_df.loc[0, column_name] = value  # Using 0 as a default index
    
    return features_df
