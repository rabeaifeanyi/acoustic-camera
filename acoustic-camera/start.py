import subprocess
import sys
import time
import signal
import webbrowser
from config import ConfigManager
import argparse
import bokeh.server.server


CONFIG_PATH = "config/config.json"
config = ConfigManager(CONFIG_PATH)

flask_process = None
bokeh_process = None

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Path to an explicit checkpoint (.keras)")
parser.add_argument("--no-flask", action="store_true", help="Start only the Bokeh app without Flask")

args, unknown = parser.parse_known_args()


def start_flask():
    """
    Starts the Flask server as a sub-process.
    """
    global flask_process
    flask_process = subprocess.Popen([sys.executable, "scripts/flask_app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Flask server started at http://127.0.0.1:5000.")


def start_bokeh():
    """
    Starts the Bokeh app as a sub-process.
    """
    global bokeh_process
    
    if args.model:
        bokeh_process = subprocess.Popen(
            [sys.executable, "-m", "bokeh", "serve", "--allow-websocket-origin=127.0.0.1:5000", "scripts/bokeh_app.py",
            "--args", "--model", args.model]
        )
    else:
        bokeh_process = subprocess.Popen(
            [sys.executable, "-m", "bokeh", "serve", "--allow-websocket-origin=127.0.0.1:5000", "scripts/bokeh_app.py", "--args"],
        )

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
    
    if args.no_flask:
        if args.model:
        
            subprocess.run([
                        sys.executable, "-m", "bokeh", "serve",
                        "scripts/bokeh_app.py",
                        "--args", "--model", args.model
                    ])
            
        else:
            
            subprocess.run([
                        sys.executable, "-m", "bokeh", "serve",
                        "scripts/bokeh_app.py"
                    ])
            
        webbrowser.open("http://localhost:5006/bokeh_app")
        
    else:

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
