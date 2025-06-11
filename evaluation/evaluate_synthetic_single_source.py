import numpy as np
import pandas as pd
from pathlib import Path
from helper_funcs import get_true_coordinates, get_id

all_measurements = {
    4 : [1, 2, 3, 4, 5, 6, 7, 8, 9],
    7 : [1, 2, 3],
    13 : [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13],
}

summary_rows = []

for group, nums in all_measurements.items():
    for num in nums:
        # Datei mit allen Terz-Bändern & a-Werten
        in_file = Path(f"Results/Reverb_Synthetic/"
                       f"predictions_id{get_id(group, num)}"
                       f"_group{group}_num{num}_all.csv")
        df = pd.read_csv(in_file)
        
        gt = get_true_coordinates(group, num)
        true_pos = np.array([gt["x_1"], gt["y_1"], gt["z_1"]])

        
        if group != 13:
            for a in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
                
                # pro Frequenz zusammenfassen
                for f in sorted(df["freq"].unique()):
                    sub = df[df["freq"] == f]
                    sub = sub[sub["a"] == a]

                    # alle x,y,z-Vorhersagen in je ein 1D-Array packen
                    xs = sub[[f"x{i}" for i in range(1,5)]].values.flatten()
                    ys = sub[[f"y{i}" for i in range(1,5)]].values.flatten()
                    zs = sub[[f"z{i}" for i in range(1,5)]].values.flatten()

                    # Mittelwert der Vorhersagen
                    x_mean = xs.mean()
                    y_mean = ys.mean()
                    z_mean = zs.mean()

                    # Abstandsfehler berechnen
                    dists    = np.sqrt((xs-true_pos[0])**2 
                                    + (ys-true_pos[1])**2 
                                    + (zs-true_pos[2])**2)
                    dists_xy = np.sqrt((xs-true_pos[0])**2 
                                    + (ys-true_pos[1])**2)

                    summary_rows.append({
                        "id":                     get_id(group, num),
                        "group":                  group,
                        "num":                    num,
                        "frequency":              f,
                        "x_mean":                 x_mean,
                        "y_mean":                 y_mean,
                        "z_mean":                 z_mean,
                        "mean_euclidean_dist":    dists.mean(),
                        "mean_euclidean_dist_xy": dists_xy.mean(),
                        "a"    : a       
                    })
                    
        else:
            a = 1.0

            
            # pro Frequenz zusammenfassen
            for f in sorted(df["freq"].unique()):
                sub = df[df["freq"] == f]

                # alle x,y,z-Vorhersagen in je ein 1D-Array packen
                xs = sub[[f"x{i}" for i in range(1,5)]].values.flatten()
                ys = sub[[f"y{i}" for i in range(1,5)]].values.flatten()
                zs = sub[[f"z{i}" for i in range(1,5)]].values.flatten()

                # Mittelwert der Vorhersagen
                x_mean = xs.mean()
                y_mean = ys.mean()
                z_mean = zs.mean()

                # Abstandsfehler berechnen
                dists    = np.sqrt((xs-true_pos[0])**2 
                                + (ys-true_pos[1])**2 
                                + (zs-true_pos[2])**2)
                dists_xy = np.sqrt((xs-true_pos[0])**2 
                                + (ys-true_pos[1])**2)

                summary_rows.append({
                    "id":                     get_id(group, num),
                    "group":                  group,
                    "num":                    num,
                    "frequency":              f,
                    "x_mean":                 x_mean,
                    "y_mean":                 y_mean,
                    "z_mean":                 z_mean,
                    "mean_euclidean_dist":    dists.mean(),
                    "mean_euclidean_dist_xy": dists_xy.mean(),
                    "a" : a
                })

# Zusammenfassung als DataFrame und CSV speichern
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("Results/evaluation_summary_Reverb_synthetic.csv",
                  index=False)
