# Source: https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/

import cv2
import numpy as np  
import glob 
import csv

# Calibration data is needed to undistort the camera image. This is not necessary for the acoustic camera, 
# but will result in a better result. 
# 
# I followed the instructions on https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/
# To use this feature, you need a printed out checkerboard pattern, a camera and images of the checkerboard 
# in different positions and angles.
# 
# In order to use this feature, set UNDISTORT to True in config/config_app.py.

path_to_images = 'camera_calibration/new_calibration_img/*.png' #TODO add the correct path to callibration images

# Define the dimensions of checkerboard 
CHECKERBOARD = (7, 10) # TODO: Change this to the dimensions of your checkerboard

# Stop the iteration when specified accuracy or max iterations are reached
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vectors for 3D and 2D points
threedpoints = [] 
twodpoints = [] 

# 3D points in real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Extracting path of individual image stored in a given directory
images = glob.glob(path_to_images)

if not images:
    print("No images found.")
    exit()

for filename in images: 
    image = cv2.imread(filename) 
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE) 
    
    # If corners are found, refine and add them to the point vectors
    if ret == True: 
        threedpoints.append(objectp3d) 
        corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria) 
        twodpoints.append(corners2) 
        image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret) 
        cv2.imshow('img', image) 
        cv2.waitKey(0) 

cv2.destroyAllWindows() 

# Perform camera calibration
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None) 

# Write results to a CSV file
with open('acoustic-camera/config/camera_calibration.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Camera Matrix"] + matrix.flatten().tolist())
    writer.writerow(["Distortion Coefficients"] + distortion.flatten().tolist())
    for i in range(len(r_vecs)):
        writer.writerow([f"Rotation Vector {i+1}"] + r_vecs[i].flatten().tolist())
        writer.writerow([f"Translation Vector {i+1}"] + t_vecs[i].flatten().tolist())

print("Kalibrierungsergebnisse wurden in 'camera_calibration.csv' gespeichert.")
