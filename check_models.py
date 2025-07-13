#!/usr/bin/env python3
"""
Check if required Ollama models are available.
If not, prompt the user to download them.
"""
import requests
import json
import sys
import subprocess
from typing import List, Dict, Optional

# Configuration
OLLAMA_API_BASE = "http://localhost:11434"
REQUIRED_MODELS = ["nomic-embed-text", "llama3:instruct"]

def get_installed_models() -> List[str]:
    """Get list of installed Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/tags")
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API: {e}")
        print("Please make sure Ollama is running (run 'ollama serve')")
        sys.exit(1)

def is_model_installed(model_name: str, installed_models: List[str]) -> bool:
    """Check if a specific model is installed."""
    return any(model.startswith(model_name) for model in installed_models)

def pull_model(model_name: str) -> bool:
    """Pull a model using the Ollama CLI."""
    print(f"\nDownloading model: {model_name}")
    print("This may take several minutes depending on your internet connection...")
    
    try:
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream the output
        for line in process.stdout:
            print(f"[Ollama] {line.strip()}")
        
        process.wait()
        return process.returncode == 0
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please install Ollama first.")
        print("Visit https://ollama.ai for installation instructions.")
        return False
    except Exception as e:
        print(f"Error pulling model: {e}")
        return False

def check_models() -> bool:
    """Check if all required models are installed."""
    print("🔍 Checking for required Ollama models...")
    installed_models = get_installed_models()
    all_installed = True
    
    for model in REQUIRED_MODELS:
        if is_model_installed(model, installed_models):
            print(f"✅ {model} is installed")
        else:
            print(f"❌ {model} is not installed")
            all_installed = False
    
    return all_installed

def main():
    """Main function to check and install required models."""
    # First check if Ollama is running
    try:
        requests.get(f"{OLLAMA_API_BASE}/api/version", timeout=5)
    except requests.exceptions.RequestException:
        print("❌ Ollama is not running. Please start Ollama first:")
        print("   ollama serve")
        sys.exit(1)
    
    # Check models
    if not check_models():
        print("\nSome required models are missing.")
        response = input("Would you like to download the missing models now? (y/n): ").strip().lower()
        
        if response == 'y':
            installed_models = get_installed_models()
            for model in REQUIRED_MODELS:
                if not is_model_installed(model, installed_models):
                    if not pull_model(model):
                        print(f"❌ Failed to download {model}")
                        sys.exit(1)
            print("\n✅ All required models have been downloaded!")
        else:
            print("\nPlease install the required models manually:")
            for model in REQUIRED_MODELS:
                print(f"   ollama pull {model}")
            sys.exit(1)
    else:
        print("\n✅ All required models are installed!")
    
    print("\nYou can now start the Research Assistant application.")

if __name__ == "__main__":
    main()
