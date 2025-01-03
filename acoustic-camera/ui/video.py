import cv2
import numpy as np
from config import load_calibration_data
from threading import Thread, Lock

# This is only used, when flask is not used

class VideoStream:
    """Class for reading video frames from a video capture object."""
    
    def __init__(self, camera_index, fps=1, desired_width=160, desired_height=120, undistort=False):
        """Initialize the video stream with the given frame dimensions and camera index."""
        self.camera_index = camera_index
        
        self.vc = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        self.vc.set(cv2.CAP_PROP_FPS, fps)
        self.vc.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        self.vc.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        
        self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    
        self.frame_height = desired_height
        self.frame_width = desired_width
        
        self.undistort = undistort
        
        self.img = np.empty((self.frame_height, self.frame_width), dtype=np.uint32)
        self.view = self.img.view(dtype=np.uint8).reshape((self.frame_height, self.frame_width, 4))[::-1, ::]
        
        self.stopped = False
        self.read_lock = Lock()
        
        self.frame = None
        
        if undistort:
            try:
                self.camera_matrix, self.dist_coeffs, _, _ = load_calibration_data('config/new_camera_calibration.csv')
                self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                    self.camera_matrix, 
                    self.dist_coeffs, 
                    (self.frame_width, self.frame_height),
                    1, # alpha
                    (self.frame_width, self.frame_height)
                )
                print(f"Camera matrix calculated, shape: {self.camera_matrix.shape}.")
            
            except FileNotFoundError:
                self.camera_matrix = None
                self.dist_coeffs = None
                self.new_camera_matrix = None
        
    def start(self):
        """Start the video capture and the frame reading thread."""
        self.stopped = False
        if not self.vc.isOpened():
            self.vc.open(self.camera_index)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
    
    def update(self):
        """Thread to read frames from the video capture."""
        while not self.stopped:
            if not self.vc.isOpened():
                continue

            rval, frame = self.vc.read()
            if rval:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                if self.undistort and self.camera_matrix is not None:
                    frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
                
                with self.read_lock:
                    self.view[:, :, 0] = frame[:, :, 2]
                    self.view[:, :, 2] = frame[:, :, 0]
                    self.view[:, :, 1] = frame[:, :, 1]
                    self.view[:, :, 3] = 255 
    
    def stop(self):
        """Stop the frame reading thread and release the video capture."""
        self.stopped = True
        self.thread.join()
        if self.vc.isOpened():
            self.vc.release()
    
    def get_frame(self):
        """Return the most recent frame."""
        with self.read_lock:
            img = self.img.copy()
        return img
