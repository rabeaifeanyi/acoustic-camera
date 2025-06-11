# -----------------------------------------------------------------------------------------------------------
# This Script generates Model results from h5 files
# -----------------------------------------------------------------------------------------------------------

from pathlib import Path
from helper_funcs import *  # type: ignore
from model_funcs import *
from config import *

# -----------------------------------------------------------------------------------------------------------
# Parameter
# -----------------------------------------------------------------------------------------------------------


signal_path ="signal_10.h5"

all_measurements = {
    "single": single_source,
    "voice": voice_sources,
    "ceiling": ceiling_mounting,
    "triple": three_sources,
    "double": two_sources,
}

multi_source_ids = {
    "triple": three_sources,
    "double": two_sources,
    "ceiling": {12 : [5, 6]}
}

# synthetic_measurements = {
#     #"double": {7 : [1, 3]}, #
#     "double_rar": {13 : [11, 13]}, #110, 112
# }

# test:

# save_path = "test_results.csv"
# get_predictions(group=4,
#                 num=1,
#                 block_size=BLOCKSIZE,
#                 num_of_blocks=NOB,
#                 frequencies=frequencies,
#                 ckpt_path=ckpt_path,
#                 signal_path="/home/rabea/Documents/Bachelorarbeit/synthetic_white_reverb.h5",
#                 save_path=save_path
#                 )

# -----------------------------------------------------------------------------------------------------------
# Script
# -----------------------------------------------------------------------------------------------------------
for category, group_dict in all_measurements.items():
    print(f"\n--- Processing category: {category} ---")
    for group, nums in group_dict.items():
        for num in nums:
            mes_id = get_id(group, num)
            print(f"Processing Group {group}, Measurement {num}")
            save_path = f"Results/{MODEL}/predictions_id{mes_id}_group{group}_num{num}_bs{BLOCKSIZE}_nob{NOB}.csv"
            main_path, _ = get_paths(group, num)
            try:
                sig_p = find_filepath(main_path, signal_path)
                
            except Exception as e:
                print(f"Failed for Group {group}, Num {num}: {e}")
                continue
        
            
            try:
                get_predictions(
                    group=group,
                    num=num,
                    block_size=BLOCKSIZE,
                    num_of_blocks=NOB,
                    frequencies=frequencies,
                    ckpt_path=ckpt_path,
                    signal_path=signal_path,
                    save_path=save_path
                )
                
            except Exception as e:
                print(f"Failed for Group {group}, Num {num}: {e}")
                
                
# # Multi sources but Sources seperate                 
# for category, group_dict in multi_source_ids.items():                
#     for group, nums in group_dict.items():
#         for num in nums:
            
#             if category == "triple":
#                 seperate_devs = [1,2,3]
#             else:
#                 seperate_devs = [1,2]
                
#             for dev in seperate_devs:
#                 p, _ = get_paths(group, num, s=dev)
                
#                 main_path, _ = get_paths(group, num, s=dev)
#                 try:
#                     sig_p = find_filepath(main_path, signal_path)
                
#                 except Exception as e:
#                     print(f"Failed for Group {group}, Num {num}: {e}, Dev {dev}")
#                     continue
                
#                 mes_id = get_id(group, num)
                
#                 print(f"Processing Group {group}, Measurement {num}")
#                 save_path = f"Results/{MODEL}/predictions_id{mes_id}_group{group}_num{num}_bs{BLOCKSIZE}_nob{NOB}_dev{dev}.csv"
                
#                 try:
#                     get_predictions(
#                         group=group,
#                         num=num,
#                         block_size=BLOCKSIZE,
#                         num_of_blocks=NOB,
#                         frequencies=frequencies,
#                         ckpt_path=ckpt_path,
#                         signal_path=signal_path,
#                         save_path=save_path,
#                         s=dev
#                     )
                    
#                 except Exception as e:
#                     print(f"Failed for Group {group}, Num {num}: {e}")
                    
                         
# for category, group_dict in synthetic_measurements.items():
#     print(f"\n--- Processing category: {category} synthetic---")
#     for group, nums in group_dict.items():
#         for num in nums:
#             print(f"Processing Group {group}, Measurement {num}")
#             save_path = f"Results/{MODEL}/predictions_id{mes_id}_group{group}_num{num}_bs{BLOCKSIZE}_nob{NOB}_synthetic.csv"
#             main_path, _ = get_paths(group, num)
#             try:
#                 sig_p = find_filepath(main_path, signal_path)
                
#             except Exception as e:
#                 print(f"Failed for Group {group}, Num {num}: {e}")
#                 continue
        
#             try:
#                 get_predictions(
#                     group=group,
#                     num=num,
#                     block_size=BLOCKSIZE,
#                     num_of_blocks=NOB,
#                     frequencies=frequencies,
#                     ckpt_path=ckpt_path,
#                     signal_path=signal_path,
#                     save_path=save_path
#                 )
                
#             except Exception as e:
#                 print(f"Failed for Group {group}, Num {num}: {e}")
                
                
                

a = get_full_signal_prediction(4, 1, BLOCKSIZE, frequencies, ckpt_path, "/home/rabea/Documents/Bachelorarbeit/Messungen/M04/M04_01/signal_10.h5")

print(a)