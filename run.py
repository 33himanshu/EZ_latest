#!/usr/bin/env python3
"""
Run script for the Research Assistant application.
Starts both the FastAPI backend and Streamlit frontend.
"""
import subprocess
import sys
import os
import signal
import time
from pathlib import Path

# Configuration
FASTAPI_CMD = ["uvicorn", "api:app", "--reload", "--port", "8000"]
STREAMLIT_CMD = ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]

# Store the process references
processes = []

def start_process(cmd, name):
    """Start a subprocess with the given command."""
    print(f"Starting {name}...")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=sys.platform == "win32"
        )
        processes.append((process, name))
        print(f"{name} started with PID {process.pid}")
        return process
    except Exception as e:
        print(f"Error starting {name}: {e}")
        return None

def log_output(process, name):
    """Log the output of a process."""
    print(f"=== {name} output ===")
    for line in process.stdout:
        print(f"[{name}] {line.strip()}")
    print(f"=== End of {name} output ===")

def cleanup():
    """Clean up all running processes."""
    print("\nShutting down processes...")
    for process, name in processes:
        if process.poll() is None:  # Process is still running
            print(f"Terminating {name} (PID: {process.pid})...")
            try:
                if sys.platform == "win32":
                    import ctypes
                    ctypes.windll.kernel32.GenerateConsoleCtrlEvent(1, process.pid)
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except Exception as e:
                print(f"Error terminating {name}: {e}")
                process.kill()
    print("All processes terminated.")

def main():
    """Main function to start the application."""
    print("🚀 Starting Research Assistant...")
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start FastAPI backend
        fastapi_process = start_process(FASTAPI_CMD, "FastAPI Backend")
        if not fastapi_process:
            print("Failed to start FastAPI backend. Exiting.")
            cleanup()
            sys.exit(1)
        
        # Give FastAPI some time to start
        print("Waiting for FastAPI to start...")
        time.sleep(3)
        
        # Start Streamlit frontend
        streamlit_process = start_process(STREAMLIT_CMD, "Streamlit Frontend")
        if not streamlit_process:
            print("Failed to start Streamlit frontend. Exiting.")
            cleanup()
            sys.exit(1)
        
        print("\n✅ Research Assistant is running!")
        print("  - Frontend: http://localhost:8501")
        print("  - API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the application.")
        
        # Log output from both processes
        while True:
            for process, name in processes:
                if process.poll() is not None:
                    print(f"{name} has stopped unexpectedly.")
                    cleanup()
                    sys.exit(1)
            time.sleep(1)
            
    except Exception as e:
        print(f"Error: {e}")
        cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
