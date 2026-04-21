#!/usr/bin/env python3
"""Convenience launcher for the HMO Chatbot.

Usage:
  python run_chatbot.py api   # FastAPI backend on :8000
  python run_chatbot.py ui    # Streamlit frontend on :8501
"""

import subprocess
import sys


def run_api() -> None:
    print("Starting FastAPI backend on http://localhost:8000")
    subprocess.run([sys.executable, "-m", "chatbot.api.main"], check=False)


def run_ui() -> None:
    print("Starting Streamlit frontend on http://localhost:8501")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "chatbot/ui/app.py"],
        check=False,
    )


if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if cmd == "api":
        run_api()
    elif cmd == "ui":
        run_ui()
    else:
        print("Usage: run_chatbot.py [api|ui]")
