import os
import acoular as ac
from pathlib import Path
from bokeh.plotting import curdoc
from ui import Dashboard
from data_processing import Processor 

from config import * 


ac.config.global_caching = "none"

CONFIG_PATH = "config/config.json"
config = ConfigManager(CONFIG_PATH)

model_name = config.get("model.name")
base_directory = Path(config.get("model.base_directory"))
config_file = config.get("model.config_file")

ckpt_directory = base_directory / model_name / config.get("model.checkpoint.directory")
file_pattern = config.get("model.checkpoint.file_pattern")
ckpt_files = ckpt_directory.glob(file_pattern)

ckpt_name = sorted(ckpt_files, key=lambda x: int(x.stem.split('-')[0]))[-1].name
ckpt_path = ckpt_directory / ckpt_name

results_folder = 'results'

# check if folder exists
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

video_index = 0
#from.helpers import list_cameras
#video_index = list_cameras()[0] # get the first valid camera index

model_on = True

device_index = get_uma16_index() # type: ignore
# device_index = 0

alphas = calculate_alphas(dx=config.get("app_settings.dx"), dz=config.get("app_settings.dz")) # type: ignore

base_path = config.get("acoular.micgeom_file.base_path")
file_name = config.get("acoular.micgeom_file.file_name")

micgeom_path = Path(ac.__file__).parent / base_path / file_name
    
processor = Processor(
    device_index,
    micgeom_path,
    results_folder,
    ckpt_path,
    model_on,
    config.get("app_default_settings.z"))

dashboard = Dashboard(
    config,
    processor,
    model_on,
    alphas)


doc = curdoc()
doc.add_root(dashboard.get_layout())
