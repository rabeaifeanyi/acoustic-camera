# -----------------------------------------------------------------------------------------------------------
# Experiment configuration for model-based acoustic source localization
#
# This script defines model paths, measurement setup parameters, and relevant configuration details 
# for evaluating neural network-based localization methods in different experimental scenarios.
#
# Core elements:
# - Model path resolution: Identifies the latest trained model checkpoint for prediction.
# - File structure setup: Includes paths for ground truth positions, measurement data, and output base directory.
# - Microphone geometry: Loads the microphone array configuration used during measurements (UMA-16 mirrored).
# - Measurement mapping: Categorizes test cases into single-, two-, and three-source experiments.
# - Frequency selection: Defines third-octave bands used for beamforming or neural network inference.
#
# Variables provided:
# - `model_dir`, `ckpt_path`: Location of the trained model and best checkpoint
# - `file_path`, `base_path`: Input/output data paths
# - `micgeom_path`, `mics`: Microphone geometry for beamforming
# - `single_source`, `two_sources`, `three_sources`, `voice_sources`, `ceiling_mounting`: 
#    Dictionaries specifying experimental setups and scenarios
# - `frequencies`, `low_freqs`, `mid_freqs`, `high_freqs`: Third-octave band categorization
# - `BLOCKSIZE`, `NOB`: Signal processing parameters for FFT and averaging
#
# This configuration is used as a foundation for both classical beamforming and deep learning evaluation pipelines.
# -----------------------------------------------------------------------------------------------------------


from pathlib import Path
import acoular as ac  # type: ignore

# -----------------------------------------------------------------------------------------------------------
# Model Configuration and Paths
# -----------------------------------------------------------------------------------------------------------

MODEL = "Reverb" #TODO "Reverb", or "Anechoic"

if MODEL == "Reverb":
    model_name = "EigmodeTransformer_Reverb" #Reverb
else:
    model_name = "EigmodeTransformer_Anechoic" #Anechoic

model_dir = Path(f"/home/rabea/Documents/Bachelorarbeit/models/{model_name}")
config_path = model_dir / 'config.toml'  # Configuration file of model (not used here directly)
ckpt_path = model_dir / 'ckpt' / 'best_ckpt'

# Find the latest checkpoint model file
ckpt_files = ckpt_path.glob('*.keras')
ckpt_name = sorted(ckpt_files, key=lambda x: int(x.stem.split('-')[0]))[-1].name
ckpt_path = model_dir / 'ckpt' / 'best_ckpt' / ckpt_name  # Full path to best checkpoint


# -----------------------------------------------------------------------------------------------------------
# File Paths, Parameters and Measurement Setup
# -----------------------------------------------------------------------------------------------------------
file_path = "Messungen.ods"  # Ground truth positions
base_path = "Measure"  # Output base path

# Microphone geometry file for simulation/processing
micgeom_path = Path(ac.__file__).parent / 'xml' / 'minidsp_uma-16_mirrored.xml'
mics = ac.MicGeom(from_file=micgeom_path)  # Load microphone geometry

single_source = {
    4  : [1, 2, 3, 4, 5, 6, 7, 8, 9], # 1 2 3 4 5 6 7 8 9 
    5  : [1, 2, 3, 5], # 10 11 13 12
    8  : [1, 2, 3, 4, 5, 6, 7, 8], # 14 15 16 17 18 19 20 21 
    9  : [1, 2, 3, 4, 5], # 22 23 24 25 26
    13 : [2, 3, 4, 5, 7, 8, 9, 10]} # 101 102 103 105 106 107 108 109

two_sources = {
    7  : [1, 2, 3],
    13 : [11, 12, 13, 15, 16]}

three_sources = {
    11 : [1, 2]}

voice_sources = {
    6  : [1, 2], 
    13 : [14]}

ceiling_mounting = {
    12 : [1, 2, 3, 4, 5, 6]}

# -----------------------------------------------------------------------------------------------------------
# Frequencies of interest, in third-octave bands 
# -----------------------------------------------------------------------------------------------------------
frequencies = [2000, 2500, 3150, 4000, 5000, 6300]

BLOCKSIZE = 256  # Number of time samples per FFT block
NOB = 64  # Number of blocks used for averaging Cross Spectral Matrix (CSM)

# Colors
lightblue="#DAE8FC"
darkblue = "#6C8EBF"
darkdarkblue = "#3C6195"

lightyellow = "#FFF2CC"
darkyellow = "#D5C165"
darkdarkyellow = "#9B8936"

lightgreen = "#D5E8D4"
darkgreen = "#82B366"
darkdarkgreen = "#48762E"

lightpurple = "#E1D5E7"
darkpurple = "#9673A6"
darkdarkpurple = "#644572"

grey = "#F5F5F5"
links = "#305B82"