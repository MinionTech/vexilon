#!/usr/bin/env python3
import subprocess
import sys
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ReloadHandler(FileSystemEventHandler):
    def __init__(self):
        self.proc = None
        self.start()

    def start(self):
        self.proc = subprocess.Popen([sys.executable, "app.py"])

    def on_modified(self, event):
        if event.src_path.endswith("app.py"):
            print("Reloading...")
            self.proc.terminate()
            self.proc.wait()
            self.start()


if __name__ == "__main__":
    handler = ReloadHandler()
    Observer().schedule(handler, ".", recursive=False).start()
    print("Watching for changes to app.py...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        handler.proc.terminate()
