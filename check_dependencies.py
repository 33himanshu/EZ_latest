#!/usr/bin/env python3
"""
Check if all required dependencies are installed.
"""
import sys
import subprocess
import pkg_resources
from typing import Dict, List, Tuple

# Required packages and their minimum versions
REQUIRED_PACKAGES = {
    'fastapi': '0.68.0',
    'uvicorn': '0.15.0',
    'python-multipart': '0.0.5',
    'pydantic': '1.8.0',
    'python-dotenv': '0.19.0',
    'PyMuPDF': '1.21.0',
    'requests': '2.26.0',
    'numpy': '1.21.0',
    'faiss-cpu': '1.7.3',
    'sentence-transformers': '2.2.2',
    'streamlit': '1.14.0',
    'streamlit-chat': '0.1.0',
    'python-magic': '0.4.27',
    'python-magic-bin': '0.4.14',
    'nltk': '3.6.3',
    'scikit-learn': '1.0.0',
    'fpdf': '1.7.2',
    'tenacity': '8.2.2',
}

def check_package(package: str, min_version: str) -> Tuple[bool, str]:
    """Check if a package is installed with the required version."""
    try:
        installed_version = pkg_resources.get_distribution(package).version
        if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
            return False, f"{package} {installed_version} (needs {min_version}+)"
        return True, f"{package} {installed_version}"
    except pkg_resources.DistributionNotFound:
        return False, f"{package} not installed"

def check_ollama() -> Tuple[bool, str]:
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"Ollama {version}"
        return False, "Ollama not found"
    except FileNotFoundError:
        return False, "Ollama not found"

def check_ollama_models() -> Tuple[bool, List[str]]:
    """Check if required Ollama models are installed."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            return False, ["Failed to list Ollama models"]
            
        installed_models = [line.split()[0] for line in result.stdout.split('\n')[1:] if line.strip()]
        required_models = ["nomic-embed-text", "llama3:instruct"]
        missing_models = [model for model in required_models if not any(m.startswith(model) for m in installed_models)]
        
        if missing_models:
            return False, [f"Missing model: {model}" for model in missing_models]
            
        return True, [f"Model found: {model}" for model in required_models]
    except Exception as e:
        return False, [f"Error checking Ollama models: {str(e)}"]

def print_status(success: bool, message: str, indent: int = 0):
    """Print a status message with appropriate formatting."""
    prefix = "  " * indent
    if success:
        print(f"{prefix}✅ {message}")
    else:
        print(f"{prefix}❌ {message}")

def main():
    """Check all dependencies and system requirements."""
    print("🔍 Checking system requirements...\n")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_ok = sys.version_info >= (3, 8)
    print_status(python_ok, f"Python {python_version} (requires 3.8+)")
    
    if not python_ok:
        print("\n❌ Please upgrade to Python 3.8 or higher.")
        sys.exit(1)
    
    # Check required Python packages
    print("\n📦 Checking Python packages...")
    all_packages_ok = True
    
    for package, min_version in REQUIRED_PACKAGES.items():
        ok, message = check_package(package, min_version)
        print_status(ok, message, indent=1)
        if not ok:
            all_packages_ok = False
    
    # Check Ollama
    print("\n🤖 Checking Ollama...")
    ollama_ok, ollama_msg = check_ollama()
    print_status(ollama_ok, ollama_msg, indent=1)
    
    # Check Ollama models if Ollama is installed
    if ollama_ok:
        models_ok, model_msgs = check_ollama_models()
        for msg in model_msgs:
            print_status(models_ok, msg, indent=2)
    else:
        models_ok = False
    
    # Print summary
    print("\n📊 Summary:")
    print(f"- Python: {'✅' if python_ok else '❌'}")
    print(f"- Packages: {'✅' if all_packages_ok else '❌'}")
    print(f"- Ollama: {'✅' if ollama_ok else '❌'}")
    print(f"- Models: {'✅' if models_ok else '❌'}")
    
    if all([python_ok, all_packages_ok, ollama_ok, models_ok]):
        print("\n🎉 All checks passed! You're ready to run the Research Assistant.")
    else:
        print("\n❌ Some requirements are not met. Please address the issues above.")
        
        # Print installation commands
        if not all_packages_ok:
            print("\nTo install missing packages, run:")
            print("  pip install -r requirements.txt")
            
        if not ollama_ok:
            print("\nTo install Ollama, visit: https://ollama.ai/")
            
        if ollama_ok and not models_ok:
            print("\nTo install required Ollama models, run:")
            print("  ollama pull nomic-embed-text")
            print("  ollama pull llama3:instruct")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
