import sounddevice as sd
import numpy as np
import csv


def get_uma16_index():
    """
    Get the index of the UMA-16 microphone array.

    Returns:
        int: Index of the UMA-16 microphone array if found, otherwise None.
    """
    devices = sd.query_devices()
    device_index = None

    for index, device in enumerate(devices):
        if "nanoSHARC micArray16" in device["name"]:
            device_index = index
            print(f"\nUMA-16 device: {device['name']} at index {device_index}\n")
            break

    if device_index is None:
        print("Could not find the UMA-16 device.")

    return device_index


def load_calibration_data(csv_file):
    """
    Load camera calibration data from a CSV file.

    This function reads a CSV file containing camera calibration data, including
    the camera matrix, distortion coefficients, rotation vectors, and translation vectors.

    Args:
        csv_file (str): Path to the CSV file containing the calibration data.

    Returns:
        tuple: 
            - numpy.ndarray: Camera matrix (3x3).
            - numpy.ndarray: Distortion coefficients.
            - list: Rotation vectors (list of 3x1 numpy arrays).
            - list: Translation vectors (list of 3x1 numpy arrays).
    """
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

        # Parse camera matrix and distortion coefficients
        camera_matrix = np.array(data[0][1:], dtype=np.float32).reshape((3, 3))
        dist_coeffs = np.array(data[1][1:], dtype=np.float32)

        # Parse rotation and translation vectors
        r_vecs = []
        t_vecs = []
        for i in range(2, len(data), 2):
            r_vec = np.array(data[i][1:], dtype=np.float32).reshape((3, 1))
            t_vec = np.array(data[i + 1][1:], dtype=np.float32).reshape((3, 1))
            r_vecs.append(r_vec)
            t_vecs.append(t_vec)

    return camera_matrix, dist_coeffs, r_vecs, t_vecs


def calculate_alphas(ratio=(4, 3), dx=None, dy=None, dz=None):
    """
    Calculate the field of view angles (alphas) based on the camera's aspect ratio and dimensions.

    Args:
        ratio (tuple): Aspect ratio of the camera (default is (4, 3)).
        dx (float, optional): Width of the field of view.
        dy (float, optional): Height of the field of view.
        dz (float): Distance from the camera to the object plane.

    Returns:
        tuple: 
            - float: Horizontal field of view angle (alpha_x) in radians.
            - float: Vertical field of view angle (alpha_y) in radians.

    Raises:
        ValueError: If neither (dx, dz) nor (dy, dz) is provided.
    """
    if dx and dz:
        alpha_x = 2 * np.arctan(dx / (2 * dz))
        alpha_y = 2 * np.arctan((ratio[1] * dx) / (2 * ratio[0] * dz))
    elif dy and dz:
        alpha_x = 2 * np.arctan((ratio[0] * dy) / (2 * ratio[1] * dz))
        alpha_y = 2 * np.arctan(dy / (2 * dz))
    else:
        raise ValueError("Either dx and dz or dy and dz must be provided.")

    return alpha_x, alpha_y


if __name__ == "__main__":
    # Example usage
    csv_file = "camera_calibration.csv"  # Replace with your actual file path
    try:
        camera_matrix, dist_coeffs, r_vecs, t_vecs = load_calibration_data(csv_file)
        print("Camera Matrix:", camera_matrix)
        print("Distortion Coefficients:", dist_coeffs)
    except FileNotFoundError:
        print(f"File {csv_file} not found.")

    # Example field of view calculation
    ratio = (4, 3)
    dx = 2.0
    dz = 5.0
    alpha_x, alpha_y = calculate_alphas(ratio=ratio, dx=dx, dz=dz)
    print(f"Alpha X: {np.degrees(alpha_x):.2f} degrees")
    print(f"Alpha Y: {np.degrees(alpha_y):.2f} degrees")
