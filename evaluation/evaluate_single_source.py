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

from helper_funcs import * 
from model_funcs import *  
from beamforming_funcs import * 
from config import *


def create_result_row_single_frequency(df_freq, group, num, frequency, signal_path, real_x, real_y, real_z, speed_of_sound, dev):
    """
    Evaluates a single frequency slice of a prediction file for one measurement.
    Returns one result row.
    """
    results = {}
    results["id"] = get_id(group, num)
    results["group"] = group
    results["num"] = num
    results["frequency"] = frequency
    results["dev"] = dev


    # Beamforming localization
    beamforming_max_x, beamforming_max_y, beamforming_max_z = single_source_beamforming(signal_path, speed_of_sound, frequency)
    results["beamforming_x"] = beamforming_max_x
    results["beamforming_y"] = beamforming_max_y
    results["beamforming_z"] = beamforming_max_z

    results["beamforming_dist"] = np.sqrt((beamforming_max_x - real_x)**2 + (beamforming_max_y - real_y)**2 + (beamforming_max_z - real_z)**2)
    results["beamforming_dist_xy"] = np.sqrt((beamforming_max_x - real_x)**2 + (beamforming_max_y - real_y)**2)
    
    # Mean of predicted positions
    xs = df_freq[[f"x{i}" for i in range(1, 5)]].values.flatten()
    ys = df_freq[[f"y{i}" for i in range(1, 5)]].values.flatten()
    zs = df_freq[[f"z{i}" for i in range(1, 5)]].values.flatten()
    
    results["x_mean"] = xs.mean()
    results["y_mean"] = ys.mean()
    results["z_mean"] = zs.mean()
    
    # Standard deviation
    center = np.array([xs.mean(), ys.mean(), zs.mean()])
    coords = np.stack((xs, ys, zs), axis=1)
    dists = np.linalg.norm(coords - center, axis=1)
    
    results["std"] = dists.std()
    results["std_xy"] = np.sqrt((xs - xs.mean())**2 + (ys - ys.mean())**2).std()
    results["std_z"] = np.sqrt((zs - zs.mean())**2).std()

    # MAE
    results["mae"] = (np.abs(xs - real_x) + np.abs(ys - real_y) + np.abs(zs - real_z)).mean() / 3 
    results["mae_xy"] = (np.abs(xs - real_x) + np.abs(ys - real_y)).mean() / 2
    
    # Distance to real coord
    results["mean_euclidean_dist"] = np.sqrt((xs - real_x)**2 + (ys - real_y)**2 + (zs - real_z)**2).mean()
    results["mean_euclidean_dist_xy"] = np.sqrt((xs - real_x)**2 + (ys - real_y)**2).mean()
    results["mean_dist_z"] = np.abs(zs - real_z).mean()
    
    # Accuracy 
    dists = [10, 20, 30, 40, 50]
    for dist in dists:
        epsilon = dist*0.01
        results[f"accuracy_{dist}cm"] = (results["mean_euclidean_dist"] <= epsilon).mean()
        results[f"accuracy_{dist}cm_xy"] = (results["mean_euclidean_dist_xy"] <= epsilon).mean()
    
    def accuracy_function(epsilon, target_accuracy=0.8):
        return (results["mean_euclidean_dist"] <= epsilon).mean() - target_accuracy

    def accuracy_function_xy(epsilon, target_accuracy=0.8):
        return (results["mean_euclidean_dist_xy"] <= epsilon).mean() - target_accuracy
    
    min_eps = 0
    max_eps = results["mean_euclidean_dist"].max()

    try:
        exact_epsilon = brentq(accuracy_function, min_eps, max_eps)
        exact_epsilon_xy = brentq(accuracy_function_xy, min_eps, max_eps)
        
        dist_with_80_accuracy = exact_epsilon * 100 
        dist_with_80_accuracy_xy = exact_epsilon_xy * 100 

        results["dist_with_80_accuracy"] = dist_with_80_accuracy
        results["dist_with_80_accuracy_xy"] = dist_with_80_accuracy_xy

    except ValueError:
        results["dist_with_80_accuracy"] = None
        results["dist_with_80_accuracy_xy"] = None
        
    return results

def evaluate_all_frequencies(file_path, group, num, signal_path, dev):
    df = pd.read_csv(file_path)
    results_all_freqs = []

    coords = get_true_coordinates(group, num)

    if coords["x_1"] != None:
        real_x = coords["x_1"]
        real_y = coords["y_1"]
        real_z = coords["z_1"]
    
    elif coords["x_2"] != None:
        real_x = coords["x_2"]
        real_y = coords["y_2"]
        real_z = coords["z_2"]
        
    elif coords["x_3"] != None:
        real_x = coords["x_3"]
        real_y = coords["y_3"]
        real_z = coords["z_3"]
        
    path, _ = get_paths(group, num, s=dev)
    signal_path = find_filepath(path, signal_path)

    #Load environment conditions
    temperature_path = find_filepath(path, "*temperature*.csv")
    if temperature_path:
        _, humidity, speed_of_sound = get_temperature(temperature_path)
    else:
        print(f"CANT FIND TEMPERATURE for group {group}, num {num}")
        speed_of_sound = 343

    for freq in df["freq"].unique():
        df_freq = df[df["freq"] == freq]
        result = create_result_row_single_frequency(df_freq, group, num, freq, signal_path, real_x, real_y, real_z, speed_of_sound, dev)
        results_all_freqs.append(result)

    return results_all_freqs


# --------------------------------------------------------------------------------
# Signal configuration for different test sets
# signal_path: standard 2-second signal
# voice_signal_path: longer signal for voice-based measurements
# --------------------------------------------------------------------------------

signal_path = "signal_10.h5"
parent_folder = "Results"

all_measurements = {
    # normal single source
    "single": single_source,
    "voice": voice_sources,
    "ceiling": {12 : [1, 2, 3, 4]},
    # all multi sources that have separate measurements for their sources
    "double" : {7 : [1, 2, 3]},
    "triple" : {11 : [1, 2]},
    "ceiling_d" : {12 : [5, 6]},
    "rar" : {13 : [11, 12, 15, 16]}
}

for MODEL in ["Anechoic", "Reverb"]:
    output_file_single_source = f"Results/evaluation_summary_{MODEL}_single_sources.csv"


    # falls file nicht existiert neue csv erstellen
    if not os.path.exists(output_file_single_source):
        print("File does not exist, creating new file")
        # Datei mit Header initialisieren
        df_empty = pd.DataFrame(columns=[
            "id", "group", "num", "frequency", "dev",
            "beamforming_x", "beamforming_y", "beamforming_z",
            "beamforming_dist", "beamforming_dist_xy",
            "x_mean", "y_mean", "z_mean",
            "std", "std_xy", "std_z",
            "mae", "mae_xy",
            "mean_euclidean_dist", "mean_euclidean_dist_xy", "mean_dist_z",
            "accuracy_10cm", "accuracy_10cm_xy",
            "accuracy_20cm", "accuracy_20cm_xy",
            "accuracy_30cm", "accuracy_30cm_xy",
            "accuracy_40cm", "accuracy_40cm_xy",
            "accuracy_50cm", "accuracy_50cm_xy",
            "dist_with_80_accuracy", "dist_with_80_accuracy_xy"
        ])
        df_empty.to_csv(output_file_single_source, index=False)



    # --------------------------------------------------------------------------------
    # Single Source
    # --------------------------------------------------------------------------------

    for category, group_dict in all_measurements.items():
        print(f"\n--- Processing category: {category} ---")
        
        if category in ["single","voice", "ceiling"]: devs = [0]
        elif category == "triple": devs = [1,2,3]
        else: devs = [1,2]
            
        for group, nums in group_dict.items():
            for num in nums:
                try:
                    print(f"Evaluating Group {group}, Measurement {num}")

                    for dev in devs:
                        # Load full prediction file
                        if len(devs) == 1:
                            file_name = f"{MODEL}/predictions_id{get_id(group, num)}_group{group}_num{num}_bs{BLOCKSIZE}_nob{NOB}.csv"
                            
                        else:
                            file_name = f"{MODEL}/predictions_id{get_id(group, num)}_group{group}_num{num}_bs{BLOCKSIZE}_nob{NOB}_dev{dev}.csv"
                            
                        file_path = os.path.join(parent_folder, file_name)

                        if not os.path.exists(file_path):
                            print(f"File not found: {file_path}")
                            continue

                        # Evaluate per frequency
                        freq_results = evaluate_all_frequencies(file_path, group, num, signal_path, dev)

                        for freq_result in freq_results:
                            pd.DataFrame([freq_result]).to_csv(output_file_single_source, mode='a', index=False, header=False)

    
                except Exception as e:
                    print(f"Failed for Group {group}, Num {num}: {e}")
                
        