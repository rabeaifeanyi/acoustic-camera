import os
from bokeh.plotting import curdoc # type: ignore
from ui import Dashboard, VideoStream
from data_processing import Processor
from config import *

# Folder for results
results_folder = 'Messungen'
# results_folder = 'path_to_results_folder' # specify the path to the results folder

# check if folder exists
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

video_index = 0
#video_index = list_cameras()[0] # get the first valid camera index

camera_on = False
model_on = True

device_index = get_uma16_index()
# device_index = 0


if camera_on:
    video_stream = VideoStream(
        video_index, 
        undistort=UNDISTORT, 
        fps=FPS, 
        desired_width=DESIRED_WIDTH, 
        desired_height=DESIRED_HEIGHT)
    
processor = Processor(
    device_index,
    micgeom_path,
    results_folder,
    ckpt_path,
    Z)

dashboard = Dashboard(
    video_stream,
    processor,
    camera_on,
    model_on,
    ESTIMATION_UPDATE_INTERVAL, 
    BEAMFORMING_UPDATE_INTERVAL,
    CAMERA_UPDATE_INTERVAL,
    THRESHOLD,
    SCALE_FACTOR,
    X,
    Y,
    Z,
    alphas,
    MIN_DISTANCE)

doc = curdoc()
doc.add_root(dashboard.get_layout())
