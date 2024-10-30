# main.py

import os
import pandas as pd
from tqdm import tqdm
from data_loader import load_json
from feature_extraction import extract_features
from tremor_extraction import extract_proximal_tremor_features

def process_json_file(json_path, output_dir, tremor_type='postural', plot=False, save_plots=False, extract_proximal=False):
    try:
        # Extract filename without extension for folder naming
        filename = os.path.splitext(os.path.basename(json_path))[0]
        patient_output_dir = os.path.join(output_dir, filename)
        plots_dir = os.path.join(patient_output_dir, 'plots')

        # Create necessary directories
        os.makedirs(plots_dir, exist_ok=True)

        # Load JSON data
        print(f"Processing {json_path}...")
        pc = load_json(json_path)

        # Extract tremor features without patient_id
        features = extract_features(
            pc=pc,
            tremor_type=tremor_type,
            plot=plot,
            save_plots=save_plots
        )

        # Optionally extract proximal tremor features (adjust if necessary)
        if extract_proximal:
            proximal_features = extract_proximal_tremor_features(
                pc, plot=plot, save_plots=save_plots
                # Remove patient_id if it's not needed
            )
            features = pd.concat([features, proximal_features], axis=1)

        # Save features to CSV
        csv_path = os.path.join(patient_output_dir, 'tremor_features.csv')
        features.to_csv(csv_path, index=False)  # Set index=False since we don't have meaningful index
        print(f"Saved features to {csv_path}")

        # Move plots to the plots directory if saving plots
        if save_plots and plot:
            for plot_file in os.listdir('.'):
                if plot_file.endswith('.svg') and filename in plot_file:
                    os.rename(plot_file, os.path.join(plots_dir, plot_file))
            print(f"Saved plots to {plots_dir}")

    except Exception as e:
        print(f"Error processing {json_path}: {e}")

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
        # Validate input and output directories
        if not os.path.isdir(input_folder):
            raise NotADirectoryError(f"Input folder '{input_folder}' does not exist or is not a directory.")
        os.makedirs(output_folder, exist_ok=True)

        # Gather all JSON files in the input folder
        json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
        if not json_files:
            print(f"No JSON files found in '{input_folder}'. Exiting.")
            return

        # Process each JSON file
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

        print("Batch processing completed successfully.")

    except Exception as e:
        print(f"Error during batch processing: {e}")

if __name__ == "__main__":
    input_folder = "/Users/Lange_L/Documents/Kinematik/Tremor2VisionPD/Data"
    output_folder = "/Users/Lange_L/Documents/Kinematik/Tremor2VisionPD/Output"
    batch_process(
        input_folder=input_folder,
        output_folder=output_folder,
        tremor_type='postural',  # oder 'kinematic'
        plot=True,               # je nach Bedarf
        save_plots=True,         # je nach Bedarf
        extract_proximal=False   # je nach Bedarf
    )
