"""
Camera Calibration Script for Acoustic Camera

This script calibrates a camera using a checkerboard pattern. 
The calibration process involves detecting corners in a checkerboard pattern from multiple images
and computing the camera matrix and distortion coefficients.

Calibration Details:
- A printed checkerboard pattern and multiple images of it taken at different angles are required.
- This calibration improves image accuracy for certain applications but is optional for the acoustic camera.

References:
- Source: https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/

Requirements:
- Images of the checkerboard pattern saved in a directory.
- Update `path_to_images` and `CHECKERBOARD` variables with appropriate values.

Output:
- The calibration results (camera matrix, distortion coefficients, rotation, and translation vectors)
  are saved to a CSV file.
  
In order to use this feature, set UNDISTORT to True in config/config_app.py.

"""

import cv2
import numpy as np  
import glob 
import csv


# Configuration
path_to_images = 'camera_calibration/new_calibration_img/*.png' #TODO add the correct path to callibration images
CHECKERBOARD = (7, 10) # TODO: Change this to the dimensions of your checkerboard


def load_images(image_path):
    """
    Load images from a directory.

    Args:
        image_path (str): Path to the images (supports glob patterns).

    Returns:
        list: List of image file paths.
    """
    images = glob.glob(image_path)
    if not images:
        print("No images found.")
        exit()
    return images


def detect_checkerboard_corners(images, checkerboard, criteria):
    """
    Detect checkerboard corners in a list of images.

    Args:
        images (list): List of image file paths.
        checkerboard (tuple): Dimensions of the checkerboard (rows, cols).
        criteria (tuple): Termination criteria for corner refinement.

    Returns:
        tuple: 3D points in real-world coordinates, 2D points in image plane.
    """
    threedpoints = []
    twodpoints = []

    objectp3d = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    for filename in images:
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray, checkerboard, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            threedpoints.append(objectp3d)
            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            twodpoints.append(refined_corners)
            image = cv2.drawChessboardCorners(image, checkerboard, refined_corners, ret)
            cv2.imshow('Checkerboard Detection', image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    return threedpoints, twodpoints


def calibrate_camera(threedpoints, twodpoints, image_size):
    """
    Perform camera calibration.

    Args:
        threedpoints (list): List of 3D points in real-world coordinates.
        twodpoints (list): List of 2D points in the image plane.
        image_size (tuple): Resolution of the images (width, height).

    Returns:
        tuple: Calibration results (camera matrix, distortion coefficients, rotation vectors, translation vectors).
    """
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, image_size, None, None
    )
    return matrix, distortion, r_vecs, t_vecs


def save_calibration_results(matrix, distortion, r_vecs, t_vecs, output_file):
    """
    Save calibration results to a CSV file.

    Args:
        matrix (np.array): Camera matrix.
        distortion (np.array): Distortion coefficients.
        r_vecs (list): Rotation vectors.
        t_vecs (list): Translation vectors.
        output_file (str): Path to the output CSV file.

    Returns:
        None
    """
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Camera Matrix"] + matrix.flatten().tolist())
        writer.writerow(["Distortion Coefficients"] + distortion.flatten().tolist())
        for i in range(len(r_vecs)):
            writer.writerow([f"Rotation Vector {i + 1}"] + r_vecs[i].flatten().tolist())
            writer.writerow([f"Translation Vector {i + 1}"] + t_vecs[i].flatten().tolist())
    print(f"Calibration results saved to '{output_file}'.")


if __name__ == "__main__":
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Load images
    images = load_images(path_to_images)

    # Detect checkerboard corners
    threedpoints, twodpoints = detect_checkerboard_corners(images, CHECKERBOARD, criteria)

    # Calibrate the camera
    gray_example = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    image_size = gray_example.shape[::-1]  # (width, height)
    matrix, distortion, r_vecs, t_vecs = calibrate_camera(threedpoints, twodpoints, image_size)

    # Save calibration results
    save_calibration_results(matrix, distortion, r_vecs, t_vecs, 'config/camera_calibration.csv')