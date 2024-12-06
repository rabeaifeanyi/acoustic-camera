import sounddevice as sd
import cv2

def get_uma16_index():
    """Get the index of the UMA-16 microphone array.
    """
    devices = sd.query_devices()
    
    device_index = None

    for index, device in enumerate(devices):
        if "nanoSHARC micArray16 UAC2.0" in device['name']:
            device_index = index
            print(f"\nUMA-16 device: {device['name']} at index {device_index}\n")
            break

    if device_index is None:
        print("Could not find the UMA-16 device.")
        
    return device_index

def list_audio_devices():
    """List all audio devices.
    """
    devices = sd.query_devices()
    
    for index, device in enumerate(devices):
        print(f"{index}: {device['name']}")

    return devices

def list_cameras():
    """List all cameras.
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
    """Test a camera.
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
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