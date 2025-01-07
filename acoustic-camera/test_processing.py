from data_processing import Processor  
import time
import acoular as ac #type:ignore
from config import ConfigManager 
from pathlib import Path
from config import * 

CONFIG_PATH = "config/config.json"
config = ConfigManager(CONFIG_PATH)

model_dir = "/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_learning_rate0.00025_epochs100_2024-10-09_09-03"
model_config_path = model_dir + "/config.toml"
ckpt_path = model_dir + '/ckpt/best_ckpt/0078-1.06.keras'
results_folder = 'results'
 
ac.config.global_caching = 'none' # type: ignore

base_path = config.get("acoular.micgeom_file.base_path")
file_name = config.get("acoular.micgeom_file.file_name")
micgeom_path = Path(ac.__file__).parent / base_path / file_name
device_index = get_uma16_index() 

model_on = True

processor = Processor(
    config=config,
    device_index=device_index,
    micgeom_path=micgeom_path,
    results_folder=results_folder,
    ckpt_path=ckpt_path,
    model_on=model_on,
    z=config.get("app_default_settings.z"),
)

#processor.log_data = True

def update_overflow_status():
    overflow = processor.dev.overflow
    print(f"Overflow Status: {overflow}")

# processor.start_beamforming()
# time.sleep(1)
# processor.stop_beamforming()

processor.start_model()
for i in range(10):
    time.sleep(1)
    update_overflow_status()
processor.stop_model()

processor.start_beamforming()
time.sleep(2)
processor.stop_beamforming()
