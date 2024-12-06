import acoular as ac
from pathlib import Path
from .funcs_camera import calculate_alphas

ac.


UNDISTORT = False # TODO set UNDISTORT to True to undistort the camera image -> callibration needed

# Speaker position
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
micgeom_path = Path(ac.__file__).parent / 'xml' / 'minidsp_uma-16_mirrored.xml' # TODO add path to different microphone geometry if needed

ckpt_path = None # TODO add path to model checkpoint if exists




