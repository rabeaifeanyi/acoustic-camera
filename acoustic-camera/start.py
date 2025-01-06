import subprocess
import sys
import time
import signal
import webbrowser
from config import ConfigManager


CONFIG_PATH = "config/config.json"
config = ConfigManager(CONFIG_PATH)

flask_process = None
bokeh_process = None


def start_flask():
    """
    Starts the Flask server as a sub-process.
    """
    global flask_process
    flask_process = subprocess.Popen([sys.executable, "skripts/flask_app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Flask server started at http://127.0.0.1:5000.")


def start_bokeh():
    """
    Starts the Bokeh app as a sub-process.
    """
    global bokeh_process
    bokeh_process = subprocess.Popen(
        [sys.executable, "-m", "bokeh", "serve", "--allow-websocket-origin=127.0.0.1:5000", "skripts/bokeh_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print("Bokeh app started at http://127.0.0.1:5006.")


def stop_processes():
    """
    Ends Flask and Bokeh processes cleanly.
    """
    global flask_process, bokeh_process

    if flask_process:
        flask_process.terminate()
        try:
            flask_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            flask_process.kill()
        print("Flask-Server beendet.")

    if bokeh_process:
        bokeh_process.terminate()
        try:
            bokeh_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            bokeh_process.kill()
        print("Bokeh-App beendet.")


def handle_exit(signal_received, frame):
    """
    Handles the termination of the main process (e.g. by Ctrl+C).
    """
    print("\nTerminate processes...")
    stop_processes()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        print("Starting Flask and Bokeh...")
        start_bokeh()

        time.sleep(2)

        start_flask()

        print("Both services are running. Press Ctrl+C to exit.")
        
        time.sleep(2)
        webbrowser.open("http://127.0.0.1:5000")
        
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"Error: {e}")
        stop_processes()
        sys.exit(1)
