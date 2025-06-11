# -----------------------------------------------------------------------------------------------------------
# Evaluation script for single-source localization results
#
# This script compares model outputs with ground truth coordinates and beamforming results.
# It calculates a range of performance metrics including accuracy, MAE, and statistical dispersion.
# The results are returned as a dictionary and can be saved for further analysis.
#
# Core steps:
# - Load metadata, signal and temperature data
# - Get ground truth positions and number of sources
# - Run beamforming-based localization for comparison
# - Compute model prediction statistics (mean error, std, accuracy thresholds)
# -----------------------------------------------------------------------------------------------------------

from pathlib import Path
from helper_funcs import * 
from model_funcs import *  
from beamforming_funcs import * 
from config import *

def create_result_row_single_frequency(df_freq, group, num, frequency, real_coords, num_sources):
    """
    Evaluates a single frequency slice of a prediction file for one measurement.
    Returns one result row.
    """
    results = {}
    results["id"] = get_id(group, num)
    results["group"] = group
    results["num"] = num
    results["frequency"] = frequency
    
    mae_values = []
    std_values = []
    
    for k in range(1, num_sources + 1):
    # Leere Liste zur Sammlung der Koordinaten, die zu Quelle k zugewiesen wurden
        x_vals, y_vals, z_vals = [], [], []

        for m in range(1, 5):  # Modellergebnisse 1 bis 4
            mask = df_freq[f"assigned_source_{m}"] == k
            x_vals.extend(df_freq.loc[mask, f"x{m}"].values)
            y_vals.extend(df_freq.loc[mask, f"y{m}"].values)
            z_vals.extend(df_freq.loc[mask, f"z{m}"].values)
        
        if len(x_vals) == 0:
            continue 

        # In Arrays umwandeln
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        z_vals = np.array(z_vals)

        # Mittelpunkt der Vorhersagen
        x_mean = np.mean(x_vals)
        y_mean = np.mean(y_vals)
        z_mean = np.mean(z_vals)

        # Ground truth
        real_x, real_y, real_z = map(float, real_coords[k - 1])

        # MAE berechnen
        mae_k = (np.abs(x_vals - real_x) + np.abs(y_vals - real_y) + np.abs(z_vals - real_z)) / 3
        mae_k = np.mean(mae_k)

        # Gesamtstreuung (STD der euklidischen Distanzen zur Mittelposition)
        center = np.array([x_mean, y_mean, z_mean])
        coords = np.stack([x_vals, y_vals, z_vals], axis=1)
        dists = np.linalg.norm(coords - center, axis=1)
        std_k = np.std(dists)

        # Speichern im results-Dict
        results[f"x{k}_mean"] = x_mean
        results[f"y{k}_mean"] = y_mean
        results[f"z{k}_mean"] = z_mean
        results[f"mae_{k}"] = mae_k
        results[f"std_{k}"] = std_k
        
        mae_values.append(mae_k)
        std_values.append(std_k)

    results["mae_avg"] = np.mean(mae_values) if mae_values else np.nan
    results["std_avg"] = np.mean(std_values) if std_values else np.nan

    return results

def evaluate_all_frequencies(file_path, group, num, signal_path, num_sources):
    df = pd.read_csv(file_path)
    results_all_freqs = []
    
    if num_sources == 2:
        coords = get_true_coordinates(group, num)
        real_coords = ((coords["x_1"], coords["y_1"], coords["z_1"]), 
                       (coords["x_2"], coords["y_2"], coords["z_2"]))
        
    elif num_sources == 3:
        if num==1: real_coords = ((-0.47, -0.01, 1.445), 
                                  (-1.122, -0.67, 1.465), 
                                  (0.345, -0.61, 1.145))
        
        else: real_coords = ((-0.47, -0.01, 1.445), 
                             (-0.882, -1.01, 1.105), 
                             (0.345, -0.61, 1.145))
        
    path, _ = get_paths(group, num)
    signal_path = find_filepath(path, signal_path)

    # Load environment conditions
    temperature_path = find_filepath(path, "*temperature*.csv")
    if temperature_path:
        _, humidity, speed_of_sound = get_temperature(temperature_path)
        print(speed_of_sound)
    else:
        print("CANT FIND TEMPERATURE")
        speed_of_sound = 343

    for freq in df["freq"].unique():
        df_freq = df[df["freq"] == freq]
        result = create_result_row_single_frequency(df_freq, group, num, freq, signal_path, real_coords, speed_of_sound, num_sources)
        results_all_freqs.append(result)

    return results_all_freqs


# --------------------------------------------------------------------------------
# Signal configuration for different test sets
# signal_path: standard 2-second signal
# voice_signal_path: longer signal for voice-based measurements
# --------------------------------------------------------------------------------

signal_path = "signal_2s_1.h5"
voice_signal_path = "signal_10.h5"

parent_folder = "Results"

# --------------------------------------------------------------------------------
# Two sources
# --------------------------------------------------------------------------------
output_file_two_sources = f"Results/evaluation_summary_two_sources.csv"
all_results_two_sources = []

num_sources = 2

for group, nums in two_sources.items():
    for num in nums:
        try:
            print(f"Evaluating Group {group}, Measurement {num}")

            # Load full prediction file
            file_name = f"predictions_group{group}_num{num}_bs{BLOCKSIZE}_nob{NOB}.csv"
            file_path = os.path.join(parent_folder, file_name)

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            # Add distances and assigned sources (in-place)
            add_dists(file_path, group, num)

            # Evaluate per frequency
            freq_results = evaluate_all_frequencies(file_path, group, num, signal_path, num_sources)

            all_results_two_sources.extend(freq_results)

        except Exception as e:
            print(f"Failed for Group {group}, Num {num}: {e}")

# Save all results to one summary file
df_summary_two_sources = pd.DataFrame(all_results_two_sources)
df_summary_two_sources.to_csv(output_file_two_sources, index=False)


# # --------------------------------------------------------------------------------
# # Three sources
# # --------------------------------------------------------------------------------
# output_file_three_sources = f"Results/evaluation_summary_three_sources.csv"
# all_results_three_sources = []
# num_sources = 3

# for group, nums in three_sources.items():
#     for num in nums:
#         try:
#             print(f"Evaluating Group {group}, Measurement {num}")

#             # Load full prediction file
#             file_name = f"predictions_group{group}_num{num}_bs{BLOCKSIZE}_nob{NOB}.csv"
#             file_path = os.path.join(parent_folder, file_name)

#             if not os.path.exists(file_path):
#                 print(f"File not found: {file_path}")
#                 continue

#             # Add distances and assigned sources (in-place)
#             add_dists(file_path, group, num)

#             # Evaluate per frequency
#             freq_results = evaluate_all_frequencies(file_path, group, num, signal_path, num_sources)

#             all_results_three_sources.extend(freq_results)

#         except Exception as e:
#             print(f"Failed for Group {group}, Num {num}: {e}")

# # Save all results to one summary file
# df_summary_three_sources = pd.DataFrame(all_results_three_sources)
# df_summary_three_sources.to_csv(output_file_three_sources, index=False)