from model_funcs import get_full_signal_prediction
from config import *
from helper_funcs import *
import os

interesting = {
    4 : [3],
    # 7 : [3],
    # 6 : [2],
}
    
FREQUENCY = 3150

res_path = f"Results/model_over_all_{MODEL}.csv"

if not os.path.exists(res_path):
        print("File does not exist, creating new file")
        # Datei mit Header initialisieren
        df_empty = pd.DataFrame(columns=[
            "id", "group", "num", "frequency", "dev",
            "model_x1", "model_y1", "model_z1",
            "model_x2", "model_y2", "model_z2",
            "model_x3", "model_y3", "model_z3",
            "model_x4", "model_y4", "model_z4"
        ])
        df_empty.to_csv(res_path, index=False)

    
for group, nums in interesting.items():
    for num in nums:
        
        path, _ = get_paths(group, num)
        signal_path = find_filepath(path, "signal_10.h5")
        dev = 1
        
        if group == 7:
            signal_path = "/home/rabea/Documents/Bachelorarbeit/Messungen/M07/M07_03/M07_03_smartphone/signal_10.h5"
            dev = 2
        
        print(signal_path)

        res = get_full_signal_prediction(group, num, [FREQUENCY], ckpt_path, signal_path)
        xs, ys, zs = res[0]
        
        num_id = get_id(group, num) 
        
        result_row = {
            "id": num_id,
            "group": group,
            "num": num,
            "frequency": FREQUENCY,
            "dev": dev, 
            "model_x1": xs[0], "model_y1": ys[0], "model_z1": zs[0],
            "model_x2": xs[1], "model_y2": ys[1], "model_z2": zs[1],
            "model_x3": xs[2], "model_y3": ys[2], "model_z3": zs[2],
            "model_x4": xs[3], "model_y4": ys[3], "model_z4": zs[3],
        }
        
        pd.DataFrame([result_row]).to_csv(res_path, mode="a", index=False, header=False)
        
        
