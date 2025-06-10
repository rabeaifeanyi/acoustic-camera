"""
Helper Script for Audio and Camera Device Management

Features:
---------
1. **Audio Devices:**
   - List all audio devices available on the system.

2. **Camera Devices:**
   - List all available camera devices.
   - Test individual cameras by displaying their feed in a window.

How to Use:
-----------
1. Run the script in a Python environment.
2. The script will:
   - Print a list of all audio devices.
   - Print a list of all available cameras.
3. If cameras are detected, it will automatically open and test each camera feed in sequence. 
   Press 'q' to quit the camera test for a specific device.

Functions:
----------
1. `list_audio_devices()`:
   - Lists all audio devices detected on the system.

2. `list_cameras()`:
   - Scans for all available cameras and lists their indices.

3. `test_camera(index)`:
   - Opens a video feed from a camera specified by its index.
   - Displays the feed in a window.
   - Press 'q' to quit the test.
"""


import sounddevice as sd
import cv2


def list_audio_devices():
    """
    List all audio devices detected on the system.

    Returns:
        list: A list of dictionaries containing details of all detected audio devices.
    """
    devices = sd.query_devices()

    for index, device in enumerate(devices):
        print(f"{index}: {device['name']}")

    return devices


def list_cameras():
    """
    List all available camera devices.

    Returns:
        list: A list of integers representing the indices of available cameras.
    """
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()

    print(f"Found {len(cameras)} cameras: {cameras}")
    return cameras


def test_camera(index):
    """
    Test a camera by displaying its video feed.

    Args:
        index (int): Index of the camera to test.

    Returns:
        None
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Camera {index} cannot be opened.")
        return

    print(f"Testing camera {index}. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"Camera {index}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Listing audio devices:")
    devices = list_audio_devices()

    print("\n\nListing cameras:")
    cams = list_cameras()

    for index in cams:
        test_camera(index)
