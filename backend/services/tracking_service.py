import subprocess

# Global process handle
tracker_process = None

def start_tracking():
    global tracker_process
    if tracker_process is None or tracker_process.poll() is not None:
        try:
            tracker_process = subprocess.Popen(['python', 'tracker/tracker_tray.py'])
            return {"status": "started"}, 200
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500
    else:
        return {"status": "already running"}, 200

def stop_tracking():
    global tracker_process
    if tracker_process and tracker_process.poll() is None:
        try:
            tracker_process.terminate()
            tracker_process = None
            return {"status": "stopped"}, 200
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500
    else:
        return {"status": "not running"}, 200
