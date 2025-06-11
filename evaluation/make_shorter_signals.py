from config import *
from helper_funcs import *

multi_source_ids = {
    "triple": three_sources,
    "double": two_sources,
    "ceiling": {12 : [5, 6]}
}


# for category, group_dict in multi_source_ids.items():            
#     for group, nums in group_dict.items():
#         for num in nums:
#             if category == "triple":
#                 seperate_devs = [1,2,3]
#             else:
#                 seperate_devs = [1,2]
               
#             for dev in seperate_devs: 
#                 p, _ = get_paths(group, num, s=dev)
               
#                 if p == None:
#                     print("not found", p)
#                     continue
                
#                 if os.path.exists(p+"signal_10.h5"):
#                     continue  
               
#                 try: 
#                     signal_path = find_filepath(p, file_name="*model_time_data.h5")
                    
                    
#                 except Exception as e:
#                     print(f"Failed for Group {group}, Num {num}: {e}, Dev {dev}")
#                     continue  
                
#                 save_shorter_signal(p[:-1], start_time=0, length=7, name=f"signal_10.h5")



# Make shorter signal
# for group, nums in two_sources.items():

#     for num in nums:
#         for dev in [1, 2]:
#             p, _ = get_paths(group, num, s=dev)
#             print(p)
#             if p == None:
#                 print(p)
#                 continue
#             signal_path = find_filepath(p, file_name="*model_time_data.h5")
#             print(signal_path)
#             save_shorter_signal(p[:-1], start_time=2, length=2, name=f"signal_2s_dev{dev}.h5")


# for group, nums in two_sources.items():
#     for num in nums:
#         for dev in [1, 2]:
#             p, _ = get_paths(group, num, s=dev)
#             #signal_path = f"signal_2s_dev{dev}.h5"
#             signal_path = f"signal_10_dev{dev}.h5"
            
#             print(p)
#             if p == None:
#                 print(p)
#                 continue
#             signal_path = find_filepath(p, file_name="*model_time_data.h5")
#             print(signal_path)
#             save_shorter_signal(p[:-1], start_time=2, length=10, name=f"signal_10_dev{dev}.h5")


# empty rooms
room_folder = "/home/rabea/Documents/Bachelorarbeit/Messungen/M00/M00_01"
room_path = "/home/rabea/Documents/Bachelorarbeit/Messungen/M00/M00_01/2024-10-25_18-56-26_model_time_data.h5"

a_folder = "/home/rabea/Documents/Bachelorarbeit/Messungen/M13/M13_01"
a_path = "/home/rabea/Documents/Bachelorarbeit/Messungen/M13/M13_01/2024-10-25_17-55-44_model_time_data.h5"

voice_folder = "/home/rabea/Documents/Bachelorarbeit/Messungen/M13/M13_14"
voice_path = "/home/rabea/Documents/Bachelorarbeit/Messungen/M13/M13_14/2024-10-28_19-42-00_model_time_data.h5"

another_folder = "/home/rabea/Documents/Bachelorarbeit/Messungen/M13/M13_05"
another = "/home/rabea/Documents/Bachelorarbeit/Messungen/M13/M13_05/2024-10-28_18-58-03_model_time_data.h5"

save_shorter_signal(another_folder, start_time=1, length=10, name=f"signal_10.h5", samples_path=another)
