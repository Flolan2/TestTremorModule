import os
import pandas as pd
from tqdm import tqdm
from pandas import json_normalize  # For flattening nested JSON structures
from data_loader import load_json
from feature_extraction import extract_features
from tremor_extraction import extract_proximal_tremor_features
import pprint  # For debugging purposes

def process_json_file(json_path, output_dir, tremor_type='postural', plot=False, save_plots=False, extract_proximal=False):
    """
    Process a single JSON file to extract tremor features and save signal data.

    Parameters:
    - json_path (str): Path to the JSON file.
    - output_dir (str): Directory where output will be saved.
    - tremor_type (str): Type of tremor to analyze ('postural' or 'kinematic').
    - plot (bool): Whether to generate plots.
    - save_plots (bool): Whether to save the generated plots.
    - extract_proximal (bool): Whether to extract proximal tremor features.
    """
    try:
        # Extract filename without extension for folder naming
        filename = os.path.splitext(os.path.basename(json_path))[0]
        patient_output_dir = os.path.join(output_dir, filename)
        plots_dir = os.path.join(patient_output_dir, 'plots')

        # Create necessary directories
        os.makedirs(plots_dir, exist_ok=True)
        print(f"[INFO] Created/Verified directory: {patient_output_dir}")

        # Load JSON data
        print(f"[INFO] Processing {json_path}...")
        pc = load_json(json_path)
        print(f"[DEBUG] Loaded data for '{filename}': Type of 'pc' is {type(pc).__name__}")
        pprint.pprint(pc)  # Print the structure for debugging

        # Save signal data to CSV
        signal_csv_path = os.path.join(patient_output_dir, 'signal_data.csv')
        try:
            if isinstance(pc, pd.DataFrame):
                pc.to_csv(signal_csv_path, index=False)
                print(f"[INFO] Saved signal data to CSV: {signal_csv_path}")
            elif isinstance(pc, dict):
                # Check for nested structures
                if any(isinstance(v, (dict, list)) for v in pc.values()):
                    try:
                        pc_df = json_normalize(pc)
                        print(f"[INFO] Flattened 'pc' with json_normalize: {pc_df.shape}")
                    except Exception as flatten_e:
                        print(f"[ERROR] Failed to flatten 'pc' using json_normalize: {flatten_e}")
                        raise
                else:
                    pc_df = pd.DataFrame(pc)
                    print(f"[INFO] Converted 'pc' dictionary to DataFrame: {pc_df.shape}")

                pc_df.to_csv(signal_csv_path, index=False)
                print(f"[INFO] Saved signal data to CSV: {signal_csv_path}")
            elif isinstance(pc, list):
                pc_df = pd.DataFrame(pc)
                pc_df.to_csv(signal_csv_path, index=False)
                print(f"[INFO] Converted 'pc' list of dictionaries to DataFrame and saved to CSV: {signal_csv_path}")
            else:
                raise ValueError(f"Unsupported 'pc' data type: {type(pc).__name__}")
        except Exception as save_e:
            print(f"[ERROR] Error saving 'signal_data.csv' for '{json_path}': {save_e}")
            raise  # Raise exception to halt processing this file

        # Extract tremor features without patient_id
        print(f"[INFO] Extracting tremor features for '{filename}'...")
        features = extract_features(
            pc=pc,
            tremor_type=tremor_type,
            plot=plot,
            save_plots=save_plots
        )
        print(f"[DEBUG] Extracted tremor features: {features.shape}")

        # Optionally extract proximal tremor features (adjust if necessary)
        if extract_proximal:
            print(f"[INFO] Extracting proximal tremor features for '{filename}'...")
            proximal_features = extract_proximal_tremor_features(
                pc, plot=plot, save_plots=save_plots
            )
            print(f"[DEBUG] Extracted proximal tremor features: {proximal_features.shape}")
            features = pd.concat([features, proximal_features], axis=1)
            print(f"[DEBUG] Combined features shape: {features.shape}")

        # Save tremor features to CSV
        csv_path = os.path.join(patient_output_dir, 'tremor_features.csv')
        features.to_csv(csv_path, index=False)
        print(f"[INFO] Saved tremor features to CSV: {csv_path}")

        # Move plots to the plots directory if saving plots
        if save_plots and plot:
            current_dir = os.getcwd()
            print(f"[DEBUG] Current working directory for plots: {current_dir}")
            for plot_file in os.listdir(current_dir):
                if plot_file.endswith('.svg') and filename in plot_file:
                    source_path = os.path.join(current_dir, plot_file)
                    destination_path = os.path.join(plots_dir, plot_file)
                    try:
                        os.rename(source_path, destination_path)
                        print(f"[INFO] Moved plot '{plot_file}' to '{plots_dir}'")
                    except Exception as move_e:
                        print(f"[ERROR] Failed to move plot '{plot_file}': {move_e}")
            print(f"[INFO] All relevant plots moved to '{plots_dir}'")

    except Exception as e:
        print(f"[ERROR] Error processing '{json_path}': {e}")

def batch_process(input_folder, output_folder, tremor_type='postural', plot=False, save_plots=False, extract_proximal=False):
    """
    Batch process all JSON files in the input folder and save results to the output folder.

    Parameters:
    - input_folder (str): Path to the input folder containing JSON files.
    - output_folder (str): Path to the output folder where results will be saved.
    - tremor_type (str): Type of tremor to analyze ('postural' or 'kinematic').
    - plot (bool): Whether to generate plots.
    - save_plots (bool): Whether to save the generated plots.
    - extract_proximal (bool): Whether to extract proximal tremor features.
    """
    try:
        # Validate input directory
        if not os.path.isdir(input_folder):
            raise NotADirectoryError(f"Input folder '{input_folder}' does not exist or is not a directory.")
        print(f"[INFO] Input directory verified: {input_folder}")

        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"[INFO] Output directory is set to: {output_folder}")

        # Gather all JSON files in the input folder
        json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
        if not json_files:
            print(f"[WARNING] No JSON files found in '{input_folder}'. Exiting.")
            return
        print(f"[INFO] Found {len(json_files)} JSON file(s) to process.")

        # Process each JSON file with a progress bar
        for json_file in tqdm(json_files, desc="Processing JSON files"):
            json_path = os.path.join(input_folder, json_file)
            process_json_file(
                json_path=json_path,
                output_dir=output_folder,
                tremor_type=tremor_type,
                plot=plot,
                save_plots=save_plots,
                extract_proximal=extract_proximal
            )

        print("[INFO] Batch processing completed successfully.")

    except Exception as e:
        print(f"[ERROR] Error during batch processing: {e}")

if __name__ == "__main__":
    # Define input and output directories
    input_folder = "/home/visualdbs/Downloads/TestTremorModule/Data"
    output_folder = "/home/visualdbs/Downloads/TestTremorModule/Output"

    # Call batch_process with desired parameters
    batch_process(
        input_folder=input_folder,
        output_folder=output_folder,
        tremor_type='postural',  # or 'kinematic'
        plot=True,               # Set to True to generate plots
        save_plots=True,         # Set to True to save plots
        extract_proximal=False   # Set to True to extract proximal tremor features
    )

