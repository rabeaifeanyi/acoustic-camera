import subprocess
import sys
import time

bokeh_process = None

def start_bokeh():
    """
    Starts the Bokeh app as a subprocess.
    """
    global bokeh_process
    bokeh_process = subprocess.Popen([sys.executable, "-m", "bokeh", "serve", "app.py", "--show"])
    print("Bokeh-App gestartet unter http://127.0.0.1:5006.")

def stop_processes():
    """
    Terminates Bokeh process.
    """
    global bokeh_process

    if bokeh_process:
        bokeh_process.terminate()
        bokeh_process.wait()
        print("Terminate Bokeh-App.")

if __name__ == "__main__":
    try:
        print("Starting Flask...")   
        start_bokeh()
        print("Both services are running. Press Ctrl+C to exit.")
        
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nEnd processes...")
        stop_processes()
        sys.exit(0)
