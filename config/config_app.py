import acoular as ac
from pathlib import Path
from .funcs_camera import calculate_alphas

UNDISTORT = False
X, Y, Z = 0.0, 0.005, 1.67 #m 

MIN_DISTANCE = 1 #m
THRESHOLD = 40
DESIRED_WIDTH = 640
DESIRED_HEIGHT = 480
FPS = 20
SCALE_FACTOR = 1.3
CAMERA_ON = False

DX, DZ = 143, 58 #m
alphas = calculate_alphas(dx=DX, dz=DZ)

# Configuration for saving results
CSV = False
H5 = False

# Update rate configurations for Interface in ms
ESTIMATION_UPDATE_INTERVAL = 100
BEAMFORMING_UPDATE_INTERVAL = 1000
CAMERA_UPDATE_INTERVAL = 100
STREAM_UPDATE_INTERVAL = 1000

# minidisps UMA-16 microphone geometry
micgeom_path = Path(ac.__file__).parent / 'xml' / 'minidsp_uma-16_mirrored.xml'

model_name = "EigmodeTransformer_learning_rate0.00025_weight_decay1e-06_epochs500_2024-10-22_10-33"
model_dir = Path(f"/home/rabea/Documents/Bachelorarbeit/models/{model_name}")
config_path = model_dir / 'config.toml'
ckpt_path = model_dir / 'ckpt' / 'best_ckpt'
ckpt_files = ckpt_path.glob('*.keras')
ckpt_name = sorted(ckpt_files, key=lambda x: int(x.stem.split('-')[0]))[-1].name
ckpt_path = model_dir / 'ckpt' / 'best_ckpt'/ ckpt_name

print(f"Using checkpoint: {ckpt_path}")