import os
from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic
from acoupipe.datasets.transfer import TransferGpuRIR
from acoupipe.datasets.utils import get_absorption_coeff

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from model_funcs import *
from modelsdfg.transformer.models import EigmodeTransformer

from pathlib import Path
import acoular as ac
import numpy as np
import pandas as pd
import tensorflow as tf
from helper_funcs import get_true_coordinates, get_id
from config import *

ckpt_path = "/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_Reverb/ckpt/best_ckpt/0478-1.68.keras"

#ckpt_path = "/home/rabea/Documents/Bachelorarbeit/models/EigmodeTransformer_Anechoic/ckpt/best_ckpt/0400-1.17.keras"


model = tf.keras.models.load_model(
    ckpt_path,
    custom_objects={"EigmodeTransformer": EigmodeTransformer},
    compile=False
)

source_positions = np.array([
    [0.0],  #  bezogen zur links rechst ausrichtung
    [0.005],  # Oben unten AUsrichtung bezogen auf den Mittelpunkt
    [1.540],  # Abstand zum Array
     
])



# all_measurements = {
#     #4  : [1],
#     4 : [1, 2, 3, 4, 5, 6, 7, 8, 9],
#     7 : [1, 2, 3],
#     13 : [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13],
#     #13: [11, 12, 13]
#     # "voice": [38,39],
#     # "ceiling": [32, 33, 34, 35],
#     # "double": [27],
#     # "triple": [29],
# }

# for group, nums in all_measurements.items():
#     for num in nums:
#         out = Path(f"Results/Reverb_Synthetic/predictions_id{get_id(group, num)}_group{group}_num{num}_bs{BLOCKSIZE}_nob{NOB}.csv")
#         synthetic_block_predictions(group, num, out, s=1)

# for group, nums in all_measurements.items():
#     for num in nums:
#         out = Path(f"Results/Reverb_Synthetic/predictions_id{get_id(group, num)}_group{group}_num{num}_all.csv")
#         synthetic_predictions_f(group, num, out, s=1)




def synthetic_predictions_f(group, num, save_path, s=1,
                          freqs=(2000, 2500, 3150, 4000, 5000, 6300)):
    gt = get_true_coordinates(group, num)
    x1, y1, z1 = gt["x_1"], gt["y_1"], gt["z_1"]
    x2, y2, z2 = gt["x_2"], gt["y_2"], gt["z_2"]
    true_positions = np.array([[x1, x2], [y1, y2], [z1, z2]])
    
    print(true_positions)

    n = true_positions.shape[1]
    realistic_walls=True
    ref=np.array([-0.021, -0.063, 0.0]) #verschiebung Referenz mic Breite, Höhe, Tiefe
    room_size=np.array([2.22, 2.12, 2.855]) #Breite, Höhe, Tiefe
    room_size = np.array([2.22, 2.12, 2.855])
    origin=np.array([1.22, 1.175,  0.76]) # bezogen auf linke untere Ecke und dann auhc Breite, Höhe, Tiefe
    
    fs=13720
    rows = []
    
    for t in [1e-4]:

        asi = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        for a in asi:
        
            #alpha = get_absorption_coeff(np.random.RandomState(2), realistic_walls=False)[:,0]
            #alpha = get_absorption_coeff(np.random.RandomState(1), realistic_walls=realistic_walls)[:, 0]

            alpha = np.ones((6,), dtype=float)*a

            mics = ac.MicGeom(from_file=Path(ac.__file__).parent / "xml" / "minidsp_uma-16_mirrored.xml")
            grid = ac.ImportGrid(pos=true_positions)

            print(mics.mpos[0][0], mics.mpos[1][0], mics.mpos[2][0])
            
            base_noise = np.eye(16) * t               # (16,16)
            noise = np.repeat(base_noise[None, ...],
                            BLOCKSIZE//2+1, axis=0)  # (F,16,16)


            trans = TransferGpuRIR(
                ref=ref,
                sample_freq=fs,
                block_size=BLOCKSIZE,
                mics=mics,
                grid=grid,
                room_size=room_size,
                alpha=alpha,
                origin=origin,
            )

            Q = np.zeros((BLOCKSIZE // 2 + 1, n, n), dtype=np.complex128)
            for i in range(n):
                Q[:, i, i] = 1.0

            model = tf.keras.models.load_model(ckpt_path)#, compile=False)

            psa = PowerSpectraAnalytic(
                Q=Q,
                transfer=trans,
                mode="wishart",
                block_size=BLOCKSIZE,
                overlap="50%",
                numsamples=int(10 * fs),
                sample_freq=fs,
                noise=noise
            )
            csm_array = psa.csm                 # (n_freqs, 16,16)
            freqs_all = psa.fftfreq()           # Vektor der Frequenz­bins

            low  = np.array(freqs) / (2**(1/6))
            high = np.array(freqs) * (2**(1/6))
            inds = {
                f_c: np.where((freqs_all >= lo) & (freqs_all < hi))[0]
                for f_c, lo, hi in zip(freqs, low, high)
            }

            
            neig = 8
            for f_c, idx in inds.items():
                if len(idx) > 1:
                    csm_band = np.mean(csm_array[idx], axis=0)
                else:
                    csm_band = csm_array[idx[0]]

                norm = csm_band[0,0]
                csm_normed = (csm_band / norm).reshape(1,16,16)

                evls, evecs = np.linalg.eigh(csm_normed)
                eigmode = evecs[..., -neig:] * evls[:, np.newaxis, -neig:]
                eigmode = np.stack([np.real(eigmode), np.imag(eigmode)], axis=3)
                eigmode = eigmode.transpose(0,2,1,3).reshape(-1, neig, 16*2)

                strength_pred, loc_pred, _ = model.predict(eigmode, verbose=0)

                row = {"id":get_id(group,num),"group":group,"num":num,"freq":f_c, "a":a, "t":t}
                for src in range(4):
                    row[f"x{src+1}"] = float(loc_pred[0,0,src])
                    row[f"y{src+1}"] = float(loc_pred[0,1,src])
                    row[f"z{src+1}"] = float(loc_pred[0,2,src])
                rows.append(row)
    
    


    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"[{group}_{num}] Dritteloktav-Bänder gesaved: {save_path}")
            


all_measurements = {
    7: [1, 3],

}

# for group, nums in all_measurements.items():
#     for num in nums:
#         out = Path(f"Results/Reverb_Synthetic/predictions_id{get_id(group, num)}_group{group}_num{num}_bs{BLOCKSIZE}_nob{NOB}.csv")
#         synthetic_block_predictions(group, num, out, s=1)

for group, nums in all_measurements.items():
    for num in nums:
        out = Path(f"Results/Reverb_Synthetic/predictions_id{get_id(group, num)}_group{group}_num{num}_all.csv")
        synthetic_predictions_f(group, num, out, s=1)


