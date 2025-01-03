import subprocess
import sys
import time
import signal
from config import ConfigManager


CONFIG_PATH = "config/config.json"
config = ConfigManager(CONFIG_PATH)

flask_process = None
bokeh_process = None

def start_flask():
    """
    Startet den Flask-Server als Subprozess.
    """
    global flask_process
    flask_process = subprocess.Popen([sys.executable, "skripts/flask_app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Flask-Server gestartet unter http://127.0.0.1:5000.")

def start_bokeh():
    """
    Startet die Bokeh-App als Subprozess.
    """
    global bokeh_process
    bokeh_process = subprocess.Popen(
        [sys.executable, "-m", "bokeh", "serve", "--allow-websocket-origin=127.0.0.1:5000", "skripts/bokeh_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print("Bokeh-App gestartet unter http://127.0.0.1:5006.")

def stop_processes():
    """
    Beendet Flask- und Bokeh-Prozesse sauber.
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
    Behandelt das Beenden des Hauptprozesses (z.B. durch Strg+C).
    """
    print("\nBeende Prozesse...")
    stop_processes()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        print("Starte Flask und Bokeh...")
        start_bokeh()

        time.sleep(2)

        start_flask()

        print("Beide Dienste laufen. Drücke Strg+C zum Beenden.")
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"Fehler: {e}")
        stop_processes()
        sys.exit(1)
