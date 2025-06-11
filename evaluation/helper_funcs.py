# -----------------------------------------------------------------------------------------------------------
# This script contains helper funcs for other scripts

# get_true_coordinates       → Loads ground truth source coordinates from setup file
# get_array_coordinates      → Loads microphone array coordinates from setup file
# get_paths                  → Finds relevant data folders for a given experiment
# find_filepath              → Recursively searches for specific files in a directory
# get_id                     → Returns the experiment ID from metadata
# get_info                   → Prints basic info about a measurement (duration, block count, etc.)
# get_save_path              → Generates standardized save paths for results
# save_shorter_signal        → Cuts and saves a shorter segment of a signal
# plot_signal                → Plots the first channel of a signal file
# save_signal_as_wav         → Saves signal as normalized mono WAV file
# count_sources              → Counts valid source coordinates in a setup
# get_temperature            → Extracts temperature, humidity, and speed of sound from CSV
# assign_points_to_sources   → Assigns predicted points to the nearest real sources
# round_csv                  → Rounds numeric values in CSV files to defined decimal places
# -----------------------------------------------------------------------------------------------------------

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from collections import defaultdict
import numpy as np #type:ignore
import glob
from matplotlib import pyplot as plt #type: ignore
import pandas as pd #type:ignore
import acoular as ac #type: ignore
import matplotlib.pyplot as plt #type: ignore
import scipy.io.wavfile as wav #type:ignore
from temperature import *


def get_true_coordinates(group, num, file_path="/home/rabea/Documents/Bachelorarbeit/Messungen.ods"):
    """
    Retrieves the true coordinates of devices for a specific experiment setup.

    Parameters
    ----------
    group : int
        The group number of the experiment.
    num : int
        The measurement number within the group.
    file_path : str
        The file path to the Excel file containing the measurement data.

    Returns
    -------
    dict
        A dictionary containing the coordinates of devices with keys 'x_1', 'y_1', 'z_1', ..., 'x_3', 'y_3', 'z_3'.
        If no data for the specified measurement is found, returns None.
        If no data for the setup is found, returns a dictionary with None values for the coordinates.
    """

    sheets = pd.read_excel(file_path, sheet_name=None, engine='odf')
    setup_df = sheets['Setup']
    exp_df = sheets['Experiments']
    
    row_exp = exp_df.loc[exp_df['Name'] == f'M{group:02}_{num:02}']
    
    if row_exp.empty:
        print(f"No data found for group {group:02} and measurement {num:02}.")
        return None
    
    setup = row_exp['Setup'].values[0]
    
    row_setup = setup_df.loc[setup_df['Name'] == setup]
    
    if row_setup.empty:
        print(f"No data found for setup {setup}.")
        return (None,) * 9 
    
    coordinates = {}

    for device in range(1, 4):
        for axis in ['x', 'y', 'z']:
            value = row_setup[f'{axis}_s{device}'].values[0]
            
            coordinates[f'{axis}_{device}'] = (None if value == '-' else value)
            
    return coordinates

def get_array_coordinates(group, num, file_path):
    """Koordinaten des Arrays einer Messung
    
    Parameters
    ----------
    group : int
        Gruppennummer der Messung
    num : int
        Nummer der Messung innerhalb der Gruppe
    file_path : str
        Pfad zur Excel-Datei mit den Messdaten
        
    Returns
    -------
    dict
        Koordinaten des Arrays als Dictionary mit den Schl sseln 'x_a', 'y_a', 'z_a'.
        Falls keine Daten f r die Messung gefunden wurden, wird None zur ckgegeben.
    """
    sheets = pd.read_excel(file_path, sheet_name=None, engine='odf')
    setup_df = sheets['Setup']
    exp_df = sheets['Experiments']
    
    row_exp = exp_df.loc[exp_df['Name'] == f'M{group:02}_{num:02}']
    
    if row_exp.empty:
        print(f"No data found for group {group:02} and measurement {num:02}.")
        return None
    
    setup = row_exp['Setup'].values[0]
    
    row_setup = setup_df.loc[setup_df['Name'] == setup]
    
    if row_setup.empty:
        print(f"No data found for setup {setup}.")
        return (None,) * 9 
    
    coordinates = {}

    for axis in ['x', 'y', 'z']:
        value = row_setup[f'{axis}_a'].values[0]
        
        coordinates[f'{axis}_a'] = (None if value == '-' else value)
            
    return coordinates
            
def get_paths(group, num, s=0):
    """
    Determines the main path and remaining paths for a given group and measurement number.

    Args:
        group (int): The group number of the measurement.
        num (int): The measurement number within the group.

    Returns:
        Tuple[str, List[str]]: A tuple containing the main path and a list of remaining paths. 
        The main path corresponds to the folder with the most devices. If no path exists, 
        returns (None, None). If no relevant folders are found, returns (base_path, []).
    """

    base_path = f"Messungen/M{group:02}/M{group:02}_{num:02}/"
    # check if path exists
    if not os.path.exists(base_path):
        return None, None
    sources = ['speaker', 'smartphone', 'ipad', 'ventilation', 'voice']
    
    
    folders = glob.glob(base_path + '*/')
    remaining_paths = []
    
    if s == 0:
        relevant_folders = []
        for folder in folders:
            
            for source in sources:
                if source in folder:
                    relevant_folders.append(folder)
        
        folders = relevant_folders
        
        if not folders:
            return base_path, []

        folder_device_map = defaultdict(list)
        for folder in folders:
            folder_name = os.path.basename(os.path.normpath(folder))
            devices = [device for device in folder_name.split('_') if device in sources]
            if devices:
                folder_device_map[folder] = devices
        
        main_path = max(folder_device_map, key=lambda k: len(folder_device_map[k]))
        main_devices = folder_device_map[main_path]
        
        # Alle anderen Pfade mit mindestens einem Gerät filtern
        remaining_paths = [path for path, devices in folder_device_map.items() if path != main_path]
    
    elif s == 1:
        main_path = f"Messungen/M{group:02}/M{group:02}_{num:02}/M{group:02}_{num:02}_speaker/"
    
    elif s == 2:
        main_path = f"Messungen/M{group:02}/M{group:02}_{num:02}/M{group:02}_{num:02}_smartphone/"
        
    elif s == 3:
        main_path = f"Messungen/M{group:02}/M{group:02}_{num:02}/M{group:02}_{num:02}_ipad/"
      
    return main_path, remaining_paths

def find_filepath(path, file_name='*_time_data.h5'):
    """
    Searches for a file with a specific pattern within a given directory path.

    Args:
        path (str): The directory path where the search is conducted.
        file_name (str): The pattern of the file name to search for. Defaults to '*_time_data.h5'.

    Returns:
        str or None: Returns the path of the first matching file found. If no file is found, returns None.

    Note:
        The search is performed recursively. If no files match the pattern in the subdirectories,
        it searches in the specified directory path. If no matching files are found, a message is printed.
    """

    files = glob.glob(os.path.join(path, '**', file_name), recursive=True)
    
    if not files:
        files = glob.glob(os.path.join(path, file_name), recursive=True)
        if not files:
            print(f"Keine {file_name}-Dateien in {path} gefunden.")
            return None
    
    return files[0]

def get_id(group, num):
    """
    Retrieves the ID of a specific experiment based on group and measurement numbers.

    Args:
        group (int): The group number of the experiment.
        num (int): The measurement number within the group.

    Returns:
        int: The ID of the experiment.
    """

    sheets = pd.read_excel("Messungen.ods", sheet_name=None, engine='odf')
    exp_df = sheets['Experiments']
    
    row_exp = exp_df.loc[exp_df['Name'] == f'M{group:02}_{num:02}']
    id = int(row_exp['id'].values[0])
    
    return id
    
def get_info(group, num, block_size=256, num_of_blocks=20):

    """
    Gibt Infos über die Messung aus

    Args:
        group (int): Gruppennummer
        num (int): Messungsnummer
        block_size (int): Blockgröße
        num_of_blocks (int): Anzahl an Blöcken

    Returns:
        None
    """
    main_path, _ = get_paths(group, num)

    if (group==8 and num==5) or (group==9 and num==4) or (group==13 and num==6):
        file = find_filepath(main_path)
        
    elif group==6 or (group==13 and num==14):
        file = find_filepath(main_path, "signal.h5")
        
    else:
        file = find_filepath(main_path, "signal_10.h5")
        
    time_samples = ac.TimeSamples(name=file)
    
    print("File:", file)
    print("Number of samples:", time_samples.numsamples)
    print("Sample frequency:", time_samples.sample_freq)
    
    duration = time_samples.numsamples/time_samples.sample_freq
    print("Duration:", duration)
    

    if duration >= 10:
        num_csms = 441000//block_size
        print("Number of calculated CSMs:", num_csms)
        
    else:
        num_csms = time_samples.numsamples//block_size
        
    total_num_predictions = num_csms//num_of_blocks
    print("Number of calculated Predictions:", total_num_predictions)    

    time_per_prediction = num_of_blocks * block_size / time_samples.sample_freq
    print("Time per Prediction:", time_per_prediction)
    print("\n")
    

def get_save_path(group, num, frequency, block_size, num_of_blocks, parent_folder):   
    """
    Generates a save path for a given experiment configuration.

    Parameters:
    group (int): Experiment group number
    num (int): Experiment number
    frequency (int or str): Frequency of the experiment
    block_size (int): Block size for processing
    num_of_blocks (int): Number of blocks for processing
    parent_folder (str): Parent folder for the save path

    Returns:
    str: A save path for the experiment configuration
    """
    if type(frequency) == int:
        folder = f"f{frequency}_bs{block_size}_nob{num_of_blocks}"
        
    else:
        folder = f"{frequency}_bs{block_size}_nob{num_of_blocks}"
    name = f"M{group:02}_{num:02}_{folder}.csv"
    save_path = os.path.join(parent_folder, folder, name)
    
    return save_path

def save_shorter_signal(path, start_time, length=10, name="signal_10.h5", samples_path=None):
    """
    Extracts a shorter segment from a signal stored in an H5 file and saves it as a new H5 file.

    Parameters
    ----------
    path : str
        Directory path where the original signal H5 file is located and where the new file will be saved.
    start_time : float
        Start time in seconds for the segment to be extracted.
    length : float, optional
        Duration in seconds of the segment to be extracted. Defaults to 10 seconds.

    Notes
    -----
    - The function extracts a segment from the specified start time with the specified length from the original signal.
    - If the specified segment is out of bounds, it prints an error message and exits.
    - The extracted signal segment is saved as 'signal_10.h5' in the specified path.
    - Plots the extracted signal segment for the first channel over time.
    """
    if samples_path is None:
        samples_path = find_filepath(path, "*time_data.h5")
    
    if samples_path is None:
        return
    
    ts = ac.TimeSamples(name=samples_path)
    
    start_sample = int(start_time * ts.sample_freq)
    end_sample = int((start_time + length) * ts.sample_freq)
    
    if start_sample >= ts.numsamples or end_sample > ts.numsamples:
        print(f"Der angeforderte Bereich (ab {start_time} s) liegt außerhalb der verfügbaren Daten.")
        return
    
    signal = ac.tools.return_result(ts, num=end_sample)[start_sample:end_sample]
    
    ts2 = ac.TimeSamples(data=signal)
    
    hd5 = ac.WriteH5(name=f'{path}/{name}', source=ts2, sample_freq=ts.sample_freq)
    
    hd5.save()
    print(f"Signal gespeichert unter: {path}")
    
    t = np.arange(start_sample, end_sample) / ts.sample_freq
    plt.figure(figsize=(14, 3))
    plt.plot(t, signal[:, 0])
    plt.ylabel('Amplitude')
    plt.xlabel('Zeit (s)')
    plt.tight_layout()
    plt.show()
    
def plot_signal(samples_path):

    """
    Plots the first channel of a signal from an H5 file.

    Parameters
    ----------
    samples_path : str
        Path to the H5 file containing the signal.

    Notes
    -----
    - The function retrieves the signal data and plots the first channel
      over time, displaying amplitude versus time in seconds.
    - If no valid signal is found, it prints an error message and exits.
    """

    ts = ac.TimeSamples(name=samples_path)
    signal = ac.tools.return_result(ts)
    
    if signal is None or signal.shape[0] == 0:
        print("Kein gültiges Signal gefunden.")
        return
    
    t = np.arange(signal.shape[0]) / ts.sample_freq  # Zeitachse in Sekunden
    
    plt.figure(figsize=(14, 3))
    plt.plot(t, signal[:, 0])  # Ersten Kanal plotten
    plt.ylabel('Amplitude')
    plt.xlabel('Zeit (s)')
    plt.title('Erstes Kanal-Signal aus H5-Datei')
    plt.grid()
    plt.tight_layout()
    plt.show()

def save_signal_as_wav(samples_path, output_path="output.wav"):
    """
    Saves the first channel of a signal from an H5 file as a WAV file.

    Parameters
    ----------
    samples_path : str
        Path to the H5 file containing the signal.
    output_path : str, optional
        Path where the WAV file will be saved. Defaults to 'output.wav'.

    Notes
    -----
    - The function extracts the first channel of the signal, normalizes it to fit
      the 16-bit integer format, and saves it as a WAV file.
    - If the signal is not valid or empty, the function prints a message and returns
      without saving a file.
    """

    ts = ac.TimeSamples(name=samples_path)
    signal = ac.tools.return_result(ts)
    
    if signal is None or signal.shape[0] == 0:
        print("Kein gültiges Signal gefunden.")
        return
    
    # Ersten Kanal extrahieren
    signal_mono = signal[:, 0]

    # Normalisieren auf 16-bit Integer-Format für WAV
    signal_mono = signal_mono / np.max(np.abs(signal_mono))  # Normierung auf [-1, 1]
    signal_mono = (signal_mono * 32767).astype(np.int16)  # In 16-bit konvertieren

    # WAV speichern
    wav.write(output_path, int(ts.sample_freq), signal_mono)

    print(f"Signal gespeichert als WAV: {output_path}")
    
def count_sources(coords):
    """Zählt die Anzahl der validen Quellen und gibt ein Tupel mit (Anzahl, Liste der vorhandenen Quellen) zurück."""
    sources = []
    for i in range(1, 4):  
        x_key = f"x_{i}"
        if x_key in coords and coords[x_key] is not None and not np.isnan(coords[x_key]):
            sources.append(i) 

    return len(sources), tuple(sources) 


def get_temperature(file_path):
    """
    Extracts temperature, humidity, and speed of sound from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the environment data.

    Returns
    -------
    temperature : float
        The temperature value from the file.
    humidity : float
        The humidity value from the file.
    speed_of_sound : float
        The speed of sound value from the file.
    """

    df = pd.read_csv(file_path)
    temperature = df["Temperature"].iloc[0]
    humidity = df["Humidity"].iloc[0]/100
    
    speed_of_sound = c0_cramer(humidity, temperature)
    
    return temperature, humidity, speed_of_sound


def assign_points_to_sources(df, real_x, real_y, real_z):
    """
    Assigns each point in the dataframe to its closest source.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the points to be assigned. Must have columns "x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4", "z1", "z2", "z3", "z4".
    real_x, real_y, real_z : array_like
        Coordinates of the sources.

    Returns
    -------
    assigned_df : pd.DataFrame
        Dataframe with the same columns as the input dataframe, plus the columns "assigned_source_1", "assigned_source_2", "assigned_source_3", "assigned_source_4", which contain the number of the closest source for each point.
    """
    nos = len(real_x)

    x_all = df[["x1", "x2", "x3", "x4"]].values
    y_all = df[["y1", "y2", "y3", "y4"]].values
    z_all = df[["z1", "z2", "z3", "z4"]].values

    dist_to_sources = np.zeros((nos, x_all.shape[0], x_all.shape[1])) 

    for i in range(nos):
        dist_to_sources[i] = np.sqrt((x_all - real_x[i])**2 + (y_all - real_y[i])**2 + (z_all - real_z[i])**2)

    closest_source = np.argmin(dist_to_sources, axis=0) + 1

    assigned_df = df.copy()
    for i in range(4):
        assigned_df[f'assigned_source_{i+1}'] = closest_source[:, i]
    
    return assigned_df


def round_csv(input_file, output_file, decimal_places=3):
    df = pd.read_csv(input_file)
    df = df.round(decimal_places)
    df.to_csv(output_file, index=False)
    print(f"Gerundete Datei gespeichert als: {output_file}")