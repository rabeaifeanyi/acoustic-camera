# -----------------------------------------------------------------------------------------------------------
# Beamforming-based localization for single source scenarios
#
# This module provides functions to compute the location of acoustic sources using beamforming 
# algorithms based on Acoular. It includes both standard delay-and-sum beamforming (BeamformerBase)
# and advanced CLEAN-SC beamforming (BeamformerCleansc), in 2D and 3D configurations.
#
# Core functionality:
# - Load microphone geometry and measurement data
# - Define spatial grid and steering vector for a given speed of sound
# - Apply either base or CLEAN-SC beamforming to determine acoustic source location
# - Extract the coordinates of the maximum beamforming level
#
# Variants included:
# - 3D and 2D localization using BeamformerBase
# - 3D and 2D localization using BeamformerCleansc
#
# Parameters:
# - signal_path: Path to the input .h5 file with multichannel time data
# - c: Speed of sound in m/s
# - frequency: Target frequency for localization in Hz
# - block_size: Number of samples per FFT block (default: 1024)
# - z: Optional fixed depth for 2D beamforming variants
#
# Returns:
# - Coordinates (x, y, z) of the beamforming maximum (estimated source position)
#
# Note: The coordinate system is flipped in x and y to match the specific mirroring
# configuration of the microphone geometry used (UMA-16 mirrored).
# -----------------------------------------------------------------------------------------------------------


import numpy as np
import acoular as ac  # type: ignore
from pathlib import Path


def single_source_beamforming(signal_path, c, frequency, block_size=256):
    """
    Computes the beamforming coordinates for a single source with given speed of sound.

    Parameters
    ----------
    signal_path : str
        The path to the signal .h5 file.
    c : float
        The speed of sound in m/s.
    frequency : float
        The frequency in Hz.

    Returns
    -------
    beamforming_max_coords : np.ndarray
        The coordinates of the beamforming maximum in meters.

    """
    mics = ac.MicGeom(from_file=Path(ac.__file__).parent / 'xml' / 'minidsp_uma-16_mirrored.xml')
    grid = ac.RectGrid3D(x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5, z_min=1, z_max=2.5, increment=0.05)
    steer = ac.SteeringVector(mics=mics, env=ac.Environment(c=c), grid=grid, steer_type='true location')
    print(steer.steer_type)
    
    time_data = ac.MaskedTimeSamples(name=signal_path)
    freq_data = ac.PowerSpectra(source=time_data, window='Hanning', overlap='50%', block_size=block_size)
    bb = ac.BeamformerBase(freq_data=freq_data, steer=steer)
    
    result = bb.synthetic(frequency, num=3)
    
    Lm_reshaped = ac.L_p(result).reshape(grid.shape)
    max_idx = np.argmax(Lm_reshaped)
    ix, iy, iz = np.unravel_index(max_idx, Lm_reshaped.shape)
    increment = 0.05
    x_coord = -1.5 + ix * increment
    y_coord = -1.5 + iy * increment
    z_coord = 1 + iz * increment
        
    return -x_coord, y_coord, z_coord

def single_source_beamforming_2D(signal_path, c, frequency, z, block_size=1024): 
    """
    Compute the beamforming coordinates for a single source in 2D with given height z.

    Parameters
    ----------
    signal_path : str
        The path to the signal .h5 file.
    c : float
        The speed of sound in m/s.
    frequency : float
        The frequency in Hz.
    z : float
        The depth of the source in meters.

    Returns
    -------
    beamforming_max_coords : np.ndarray
        The coordinates of the beamforming maximum in meters.

    """
    mics = ac.MicGeom(from_file=Path(ac.__file__).parent / 'xml' / 'minidsp_uma-16_mirrored.xml')
    grid = ac.RectGrid(x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5, z=z, increment=0.05)
    steer = ac.SteeringVector(mics=mics, env=ac.Environment(c=c), grid=grid)
    
    time_data = ac.MaskedTimeSamples(name=signal_path)
    freq_data = ac.PowerSpectra(source=time_data, window='Hanning', overlap='50%', block_size=block_size)
    bb = ac.BeamformerBase(freq_data=freq_data, steer=steer)
    
    result = bb.synthetic(frequency, num=3)
        
    Lm = ac.L_p(result)[::-1, :]
    beamforming_max_coords = np.array(np.unravel_index(np.argmax(Lm), Lm.shape)) * 0.05 - 1.5
      
    return beamforming_max_coords


def single_source_cleansc(signal_path, c, frequency, block_size=1024):
    """
    Computes the beamforming coordinates for a single source with given speed of sound.

    Parameters
    ----------
    signal_path : str
        The path to the signal .h5 file.
    c : float
        The speed of sound in m/s.
    frequency : float
        The frequency in Hz.

    Returns
    -------
    beamforming_max_coords : np.ndarray
        The coordinates of the beamforming maximum in meters.

    """
    mics = ac.MicGeom(from_file=Path(ac.__file__).parent / 'xml' / 'minidsp_uma-16_mirrored.xml')
    grid = ac.RectGrid3D(x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5, z_min=1, z_max=2.5, increment=0.1)
    steer = ac.SteeringVector(mics=mics, env=ac.Environment(c=c), grid=grid)
    
    time_data = ac.MaskedTimeSamples(name=signal_path)
    freq_data = ac.PowerSpectra(source=time_data, window='Hanning', overlap='50%', block_size=block_size)
    bb_cleansc = ac.BeamformerCleansc(freq_data=freq_data, steer=steer)
    
    result = bb_cleansc.synthetic(frequency, num=3)
    
    Lm_reshaped = ac.L_p(result).reshape(grid.shape)
    max_idx = np.argmax(Lm_reshaped)
    ix, iy, iz = np.unravel_index(max_idx, Lm_reshaped.shape)
    x_coord = -1.5 + ix * 0.1
    y_coord = -1.5 + iy * 0.1
    z_coord = 1 + iz * 0.1
        
    return -x_coord, -y_coord, z_coord

def single_source_cleansc_2D(signal_path, c, frequency, z, block_size=1024): 
    """
    Compute the beamforming coordinates for a single source in 2D with given height z.

    Parameters
    ----------
    signal_path : str
        The path to the signal .h5 file.
    c : float
        The speed of sound in m/s.
    frequency : float
        The frequency in Hz.
    z : float
        The depth of the source in meters.

    Returns
    -------
    beamforming_max_coords : np.ndarray
        The coordinates of the beamforming maximum in meters.

    """
    mics = ac.MicGeom(from_file=Path(ac.__file__).parent / 'xml' / 'minidsp_uma-16_mirrored.xml')
    grid = ac.RectGrid(x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5, z=z, increment=0.1)
    steer = ac.SteeringVector(mics=mics, env=ac.Environment(c=c), grid=grid)
    
    time_data = ac.MaskedTimeSamples(name=signal_path)
    freq_data = ac.PowerSpectra(source=time_data, window='Hanning', overlap='50%', block_size=block_size)

    bb_cleansc = ac.BeamformerCleansc(freq_data=freq_data, steer=steer)
    
    result = bb_cleansc.synthetic(frequency, num=3)
        
    Lm = ac.L_p(result)[::-1, :]
    beamforming_max_coords = np.array(np.unravel_index(np.argmax(Lm), Lm.shape)) * 0.1 - 1.5
      
    return beamforming_max_coords
