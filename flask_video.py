from flask import Flask, Response
import cv2
from config import load_calibration_data

# http://localhost:5000/video_feed

app = Flask(__name__)

camera_index = 0
camera = cv2.VideoCapture(camera_index)

def gen_undistorted_frames():
    frame_width, frame_height = 640, 480
    
    try: 
        camera_matrix, dist_coeffs, _, _ = load_calibration_data('config/camera_calibration.csv')
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, 
            dist_coeffs, 
            (frame_width, frame_height),
            1, # alpha
            (frame_width, frame_height)
        )
        print(f"Camera matrix calculated, shape: {camera_matrix.shape}.")
    
    except FileNotFoundError:
        camera_matrix = None
        dist_coeffs = None
        new_camera_matrix = None
        print("Calibration file not found.")
    
    while True:
        success, frame = camera.read()
        
        if not success:
            break
        else:
            if camera_matrix is not None:
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

            ret, buffer = cv2.imencode('.jpg', frame)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', port=5000)