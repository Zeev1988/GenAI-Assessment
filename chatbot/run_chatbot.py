#!/usr/bin/env python3
"""
Convenience launcher for the HMO Chatbot (Part 2).

Usage
-----
  python run_chatbot.py api      # Start the FastAPI backend  (port 8000)
  python run_chatbot.py ui       # Start the Streamlit frontend (port 8501)
  python run_chatbot.py both     # Start both in parallel (requires two terminals
                                 # or use the --background flag)

You almost always want to run `api` and `ui` in two separate terminals.
"""

import subprocess
import sys
import time


def run_api() -> None:
    print("Starting FastAPI backend on http://localhost:8000 …")
    print("Interactive docs: http://localhost:8000/docs")
    print("Press Ctrl+C to stop.\n")
    subprocess.run(
        [sys.executable, "-m", "chatbot.api.main"],
        check=False,
    )


def run_ui() -> None:
    print("Starting Streamlit frontend on http://localhost:8501 …")
    print("Press Ctrl+C to stop.\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "chatbot/ui/app.py"],
        check=False,
    )


def usage() -> None:
    print(__doc__)
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()

    cmd = sys.argv[1].lower()
    if cmd == "api":
        run_api()
    elif cmd == "ui":
        run_ui()
    elif cmd == "both":
        import threading
        t = threading.Thread(target=run_api, daemon=True)
        t.start()
        time.sleep(3)
        run_ui()  # blocks in the main thread
    else:
        usage()
