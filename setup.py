#!/usr/bin/env python3
"""
Setup script for the Research Assistant application.
This script helps set up the required environment and dependencies.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

# Configuration
REQUIRED_PYTHON = (3, 10)
REQUIREMENTS_FILE = "requirements.txt"
ENV_FILE = ".env"
ENV_EXAMPLE_FILE = ".env.example"
UPLOAD_DIR = "uploads"
VECTOR_STORE_DIR = "vector_store"

def check_python_version():
    """Check if the Python version is sufficient."""
    if sys.version_info < REQUIRED_PYTHON:
        print(f"Error: Python {'.'.join(map(str, REQUIRED_PYTHON))} or higher is required.")
        print(f"You are using Python {sys.version.split()[0]}.")
        sys.exit(1)

def create_virtualenv():
    """Create a Python virtual environment if it doesn't exist."""
    venv_dir = "venv"
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
            print(f"Virtual environment created at {venv_dir}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            return False
    else:
        print(f"Virtual environment already exists at {venv_dir}")
        return True

def install_dependencies():
    """Install required Python packages."""
    print("Installing dependencies...")
    pip_cmd = [
        "venv/bin/pip" if os.name != 'nt' else "venv\\Scripts\\pip",
        "install",
        "-r",
        REQUIREMENTS_FILE
    ]
    
    try:
        subprocess.run(pip_cmd, check=True)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def setup_environment():
    """Set up the environment configuration."""
    if not os.path.exists(ENV_FILE):
        if os.path.exists(ENV_EXAMPLE_FILE):
            print(f"Creating {ENV_FILE} from example...")
            with open(ENV_EXAMPLE_FILE, 'r') as src, open(ENV_FILE, 'w') as dst:
                dst.write(src.read())
            print(f"Created {ENV_FILE}. Please review and edit the configuration as needed.")
        else:
            print(f"Warning: {ENV_EXAMPLE_FILE} not found. Please create a .env file manually.")
    else:
        print(f"{ENV_FILE} already exists. Skipping creation.")

def create_directories():
    """Create necessary directories."""
    for directory in [UPLOAD_DIR, VECTOR_STORE_DIR]:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")

def print_next_steps():
    """Print the next steps for the user."""
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Make sure Ollama is installed and running")
    print("2. Pull the required models:")
    print("   ollama pull nomic-embed-text")
    print("   ollama pull llama3:instruct")
    print("3. Start the Ollama server:")
    print("   ollama serve")
    print("4. In a new terminal, activate the virtual environment:")
    if platform.system() == "Windows":
        print("   .\\venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("5. Start the FastAPI backend:")
    print("   uvicorn api:app --reload --port 8000")
    print("6. In another terminal, start the Streamlit frontend:")
    print("   streamlit run streamlit_app.py")
    print("\nThe application will be available at http://localhost:8501")

def main():
    """Main setup function."""
    print("🚀 Setting up Research Assistant...")
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    if not create_virtualenv():
        print("Failed to create virtual environment.")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies.")
        sys.exit(1)
    
    # Set up environment
    setup_environment()
    
    # Create necessary directories
    create_directories()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
