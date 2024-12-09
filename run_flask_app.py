import subprocess
import sys
import time

#TODO add flask app

flask_process = None
bokeh_process = None

def start_flask():
    """
    Starts the Flask server as a subprocess.    
    """
    global flask_process
    flask_process = subprocess.Popen([sys.executable, "flask_video.py"])
    print("Flask-Server gestartet unter http://127.0.0.1:5000.")

def start_bokeh():
    """
    Starts the Bokeh app as a subprocess.
    """
    global bokeh_process
    bokeh_process = subprocess.Popen([sys.executable, "-m", "bokeh", "serve", "app.py", "--show"])
    print("Bokeh-App gestartet unter http://127.0.0.1:5006.")

def stop_processes():
    """
    Terminates Flask and Bokeh processes.
    """
    global flask_process, bokeh_process

    if flask_process:
        flask_process.terminate()
        flask_process.wait()
        print("Terminate Flask-Server.")

    if bokeh_process:
        bokeh_process.terminate()
        bokeh_process.wait()
        print("Terminate Bokeh-App.")

if __name__ == "__main__":
    try:
        print("Start Flask and Bokeh...")   

        start_flask()
        time.sleep(2)
        start_bokeh()
        print("Both services are running. Press Ctrl+C to exit.")
        
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nEnd processes...")
        stop_processes()
        sys.exit(0)
