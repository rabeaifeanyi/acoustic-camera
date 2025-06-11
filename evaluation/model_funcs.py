# -----------------------------------------------------------------------------------------------------------
# Model prediction and evaluation pipeline for acoustic source localization
#
# This script handles loading a trained deep learning model and applying it to measurement data 
# to predict source locations and strengths across third-octave frequency bands. It also includes 
# functions for evaluation against ground truth data.
#
# Core components:
# - `ModelSetup`: Class for preparing input features (CSMs → eigmodes) and running model predictions
# - `get_predictions`: Executes prediction in batches, aggregates results and saves them as CSV
# - `add_dists`: Adds evaluation metrics to the CSV, including distances to ground truth and per-source error
#
# Key processing steps:
# - Load time signal data and compute FFT + Cross Spectral Matrices
# - Average CSMs across blocks and convert to eigmode representation
# - Run deep learning model to predict source locations and strengths
# - Post-process predictions: mirror, scale, normalize
# - Match predictions to closest ground truth sources and compute evaluation metrics
#
# Outputs:
# - A CSV file with predicted source coordinates and strength per frequency band
# - Distance and deviation statistics with respect to ground truth positions
#
# Dependencies:
# - TensorFlow, Acoular, NumPy, SciPy, Pandas
# - Custom utility functions loaded from `helper_funcs.py`
#
# This script supports real-world evaluation of learned source localization models in structured 
# experimental setups using third-octave band decomposition.
# -----------------------------------------------------------------------------------------------------------


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import acoular as ac  # type: ignore
from scipy.optimize import brentq  # type: ignore
from helper_funcs import *  # type: ignore
from config import *


ac.config.global_caching = 'none' 

# -----------------------------------------------------------------------------------------------------------
# Model Wrapper Class
# -----------------------------------------------------------------------------------------------------------
class ModelSetup:
    """
    This class handles loading the model, generating the CSM (Cross Spectral Matrix),
    transforming it into eigmodes, and making predictions for third-octave frequency bands.
    """
    def __init__(self, file, block_size, frequencies, ckpt_path, mode=3):
        # Load time-domain microphone signals
        time_samples = ac.TimeSamples(file=file)

        # Preprocessing: scale signals using SourceMixer (normalize amplitude)
        source_mixer = ac.SourceMixer(sources=[time_samples], weights=np.array([1/0.0016]))

        # Compute FFT (Fast Fourier Transform) blocks
        fft = ac.RFFT(source=source_mixer, block_size=block_size)
        self.csm_gen = ac.CrossPowerSpectra(source=fft)  # Generator for Cross-Spectral Matrices (CSMs)
        self.freqs = fft.fftfreq()  # Get FFT frequency bins

        # Define index ranges for each third-octave band
        if mode == 3:
            self.inds = {
                f_c: np.where(
                    (self.freqs >= f_c / (2**(1/6))) & 
                    (self.freqs <  f_c * (2**(1/6)))
                )[0] for f_c in frequencies
            }
        
        # Or set indices for single frequency or list of indices    
        elif mode == 1:
            self.inds = {
                f_c: [np.searchsorted(self.freqs, f_c)]
                for f_c in frequencies
            }

        # Load trained deep learning model (Keras format)
        self.model = tf.keras.models.load_model(ckpt_path)
        self.csm_shape = (int(block_size/2+1), 16, 16)  # Shape of full CSM
        self.sample_freq = time_samples.sample_freq  # Sampling rate in Hz
        self.duration = time_samples.numsamples / self.sample_freq  # Duration of signal in seconds
        self.numsamples = time_samples.numsamples
        self.num_blocks = int(self.duration * self.sample_freq // block_size)

    def get_gen(self):
        """Returns the generator that yields Cross Spectral Matrices (CSMs)."""
        return self.csm_gen.result(num=1)

    def get_csm_list(self, gen, num_of_blocks):
        """Collect a list of CSMs for averaging."""
        return [next(gen) for _ in range(num_of_blocks)]

    def _preprocess_csm(self, csm):
        """
        Normalize and convert CSM to eigmode representation.
        Steps:
        1. Normalize by the (0,0) component
        2. Eigen-decomposition
        3. Form eigmode tensor
        """
        csm_norm = csm[0, 0]  # normalization scalar
        csm = csm / csm_norm  # normalized CSM
        csm = csm.reshape(1, 16, 16)
        
        neig = 8  # number of eigenmodes to use
        evls, evecs = np.linalg.eigh(csm)
        eigmode = evecs[..., -neig:] * evls[:, np.newaxis, -neig:]  # Shape: (1, 16, 8)
        eigmode = np.stack([np.real(eigmode), np.imag(eigmode)], axis=3)  # Shape: (1, 16, 8, 2)
        eigmode = np.transpose(eigmode, [0, 2, 1, 3])  # Shape: (1, 8, 16, 2)
        eigmode = eigmode.reshape(-1, eigmode.shape[1], eigmode.shape[2]*eigmode.shape[3])  # (1, 8, 32)

        return eigmode, csm_norm

    def _preprocess_csms(self, csm_mean):
        """
        Compute third-octave averaged CSMs and convert them to eigmodes for each band.
        """
        eigmodes, csm_norms = [], []
        for f_ind in self.inds.values():
            if len(f_ind) > 1:
                csm_third = np.mean(csm_mean[f_ind[0]:f_ind[-1]], axis=0)  # Average CSM over band
            else:
                csm_third = csm_mean[f_ind[0]]

            csm_third = csm_third.reshape(16, 16)
            eigmode, csm_norm = self._preprocess_csm(csm_third)
            eigmodes.append(eigmode)
            csm_norms.append(csm_norm)

        eigmodes_array = np.stack(eigmodes, axis=0)  # (num_bands, 1, 8, 32)
        s = eigmodes_array.shape
        eigmodes_array = eigmodes_array.reshape(s[0], s[2], s[3])  # (num_bands, 8, 32)
        csm_norms_array = np.stack(csm_norms, axis=0) 

        return eigmodes_array, csm_norms_array

    def get_prediction(self, csm_list):
        """
        Make model predictions from a list of CSMs.
        Returns:
        - Source locations: shape (num_bands, 3, num_sources)
        - Source strengths (in dB)
        """
        csm_mean = np.mean(csm_list, axis=0)  # average CSM over blocks
        csm_mean = csm_mean.reshape(self.csm_shape)


        eigmodes, csm_norms = self._preprocess_csms(csm_mean)
        
        strength_pred, loc_pred, _ = self.model.predict(eigmodes, verbose=0) # (num_bands, 4), (num_bands, 3, 4)

        # Apply normalization scaling to source strengths
        strength_pred *= np.real(csm_norms)[:, np.newaxis]  # Scale prediction with original energy
        strength_pred = ac.L_p(strength_pred)  # Convert to decibel: L_p = 10 * log10(p^2 / p0^2)

        # Apply spatial scaling and mirroring to get real-world coordinates
        #loc_pred *= np.array([1.0, 1.0, 0.5])[np.newaxis, :, np.newaxis]  # Scale z-axis (depth)
        #loc_pred -= np.array([0.0, 0.0, -1.5])[np.newaxis, :, np.newaxis]  # Translate z
        loc_pred[:, 0, :] *= -1  # Mirror x-axis

        return loc_pred, strength_pred


# -----------------------------------------------------------------------------------------------------------
# Batch Prediction Function
# -----------------------------------------------------------------------------------------------------------
def get_predictions(group, num, block_size, num_of_blocks, frequencies, ckpt_path, signal_path, save_path, s=0):
    mes_id = get_id(group, num)
    
    if s>=1:
        main_path, _ = get_paths(group, num, s=s)
    else:
        main_path, _ = get_paths(group, num)
    
    signal_path = find_filepath(main_path, signal_path)  # Path to time signal file

    model_setup = ModelSetup(signal_path, block_size, frequencies, ckpt_path)
    gen = model_setup.get_gen()

    results = []
    block_counter = 0
    frequencies = list(model_setup.inds.keys())  # Ensure consistent frequency ordering

    while True:
        try:
            data = model_setup.get_csm_list(gen, num_of_blocks)
            loc_pred, strength_pred = model_setup.get_prediction(data)
            block_counter += 1

            for i, f_c in enumerate(frequencies):
                row = [mes_id, group, num, block_counter, f_c]
                for j in range(4):  # 4 strongest predicted sources
                    x = loc_pred[i, 0, j]
                    y = loc_pred[i, 1, j]
                    z = loc_pred[i, 2, j]
                    s = strength_pred[i, j]
                    row.extend([x, y, z, s])
                results.append(row)

        except StopIteration:
            print("Generator exhausted.")
            break

    # Save results
    coord_cols = sum([[f"x{i+1}", f"y{i+1}", f"z{i+1}", f"s{i+1}"] for i in range(4)], [])
    columns = ["id", "group", "num", "block", "freq"] + coord_cols
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")
    
    
def get_full_signal_prediction(group, num, frequency, ckpt_path, signal_path, s=0, block_size=1024):
    """
    Compute source predictions over the entire signal ("langsame Variante").
    Returns only the coordinates of the four strongest sources for each frequency band.

    Parameters:
        group (str): measurement group identifier
        num (int): measurement number
        block_size (int): size of FFT blocks
        frequencies (list): list of center frequencies (third-octave bands)
        ckpt_path (str): path to the trained Keras model
        signal_path (str): relative path or pattern to the time-signal file
        s (int, optional): session index for hierarchical paths

    Returns:
        loc_pred_full (np.ndarray): array of shape (num_bands, 3, 4) containing x, y, z coords
                                    for the four strongest sources per band.
    """
    # Determine file paths
    mes_id = get_id(group, num)

    # Initialize model setup
    model_setup = ModelSetup(signal_path, block_size, frequency, ckpt_path)
    gen = model_setup.get_gen()

    # Determine total number of blocks from the signal duration
    total_blocks = model_setup.num_blocks
    # print(f"Total number of blocks: {total_blocks}")

    # Read all CSM blocks at once for the full signal
    csm_list = model_setup.get_csm_list(gen, total_blocks)
    # print(len(csm_list))

    # Run prediction over the averaged CSMs for each frequency band
    loc_pred_full, _ = model_setup.get_prediction(csm_list)

    return loc_pred_full
