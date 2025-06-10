from flask import Flask, render_template, Response
import cv2
from bokeh.embed import server_document
import json
import os


app = Flask(__name__,
            template_folder=os.path.abspath("templates"))

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    bokeh_script = server_document("http://localhost:5006/bokeh_app")
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    return render_template('index.html', bokeh_script=bokeh_script, json_data=config)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

