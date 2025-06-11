import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from config import *
from helper_funcs import *
import acoular as ac

import sys
sys.path.insert(0, '/home/eschenhagen/acoupipe/src')
from acoupipe.datasets.transfer import TransferISM #/home/eschenhagen/acoupipe/src/acoupipe/datasets/transfer.py
from acoular import MicGeom
import numpy as np

multi_source_ids = {
    27: (7, 1),
    29: (7, 3),
}

multi_source_ids_a = {
    110: (13, 11),
    112: (13, 13),
}


for group, num in multi_source_ids_a.values():
    coords = get_true_coordinates(group, num)
    source1 = [coords["x_1"], coords["y_1"], coords["z_1"]]
    source2 = [coords["x_2"], coords["y_2"], coords["z_2"]]
    
    mics = ac.MicGeom(from_file=Path(ac.__file__).parent / 'xml' / 'minidsp_uma-16.xml')
    room_size = np.array([2.13, 2.86, 2.22])
    origin = np.array([1.22, 0.628,1.15])
    alpha = np.ones((6,)) * 0.2  
    
    sfreq = 22050
    duration = 10
    num_samples = duration * sfreq
    

    # ohne reverb
    n1 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=1)
    n2 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=2)
    
    p1 = ac.PointSource(signal=n1, mics=mics, loc=(coords["x_1"], coords["y_1"], coords["z_1"]))
    p2 = ac.PointSource(signal=n2, mics=mics, loc=(coords["x_2"], coords["y_2"], coords["z_2"]))
    
    
    h5savefile = Path(f'/home/rabea/Documents/Bachelorarbeit/Messungen/M{group:02}/M{group:02}_{num:02}/M{group:02}_{num:02}_speaker_and_smartphone/signal_10_synthetic_reverb.h5')
    
    p = ac.Mixer(source=p1, sources=[p2])
    wh5 = ac.WriteH5(source=p, file=h5savefile)
    
    wh5.save()
    