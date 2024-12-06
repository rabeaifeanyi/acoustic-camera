import numpy as np
import csv


def load_calibration_data(csv_file):
    """
    Load camera calibration data from a CSV file.
    This is turned off by default. To enable it, set UNDISTORT to True in config/config_app.py.
    """
    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        
        camera_matrix = np.array(data[0][1:], dtype=np.float32).reshape((3, 3))
        
        dist_coeffs = np.array(data[1][1:], dtype=np.float32)
        
        # Rotation and Translation Vectors
        r_vecs = []
        t_vecs = []
        for i in range(2, len(data), 2):
            r_vec = np.array(data[i][1:], dtype=np.float32).reshape((3, 1))
            t_vec = np.array(data[i+1][1:], dtype=np.float32).reshape((3, 1))
            r_vecs.append(r_vec)
            t_vecs.append(t_vec)
    
    return camera_matrix, dist_coeffs, r_vecs, t_vecs

def calculate_alphas(ratio=(4, 3), dx=None, dy=None, dz=None):    

    if dx and dz:
        alpha_x = 2 * np.arctan(dx / (2 * dz))
        alpha_y = 2 * np.arctan((ratio[1] * dx) / (2 * ratio[0] * dz))
        
    elif dy and dz:
        alpha_x = 2 * np.arctan((ratio[0] * dy) / (2 * ratio[1] * dz))
        alpha_y = 2 * np.arctan(dy / (2 * dz))

    else:
        raise ValueError("Either dx and dz or dy and dz must be provided.")
        
    return alpha_x, alpha_y