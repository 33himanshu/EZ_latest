#!/usr/bin/env python3
"""
Clean up the application data.
This script removes the vector store and uploads directory.
"""
import shutil
import os
import sys
from pathlib import Path

def confirm(prompt: str) -> bool:
    """Ask for confirmation before proceeding."""
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        if response in ('y', 'yes'):
            return True
        elif response in ('n', 'no'):
            return False
        print("Please enter 'y' or 'n'.")

def clean_directory(directory: Path, description: str) -> bool:
    """Remove a directory and its contents."""
    if not directory.exists():
        print(f"{description} does not exist.")
        return False
    
    try:
        shutil.rmtree(directory)
        print(f"✅ Deleted {description}.")
        return True
    except Exception as e:
        print(f"❌ Error deleting {description}: {e}")
        return False

def main():
    """Main function to clean up application data."""
    print("🧹 Research Assistant Cleanup Tool\n")
    
    # Define directories to clean
    directories = {
        Path("vector_store"): "Vector store",
        Path("uploads"): "Uploads directory"
    }
    
    # Check which directories exist
    dirs_to_clean = [
        (path, desc) for path, desc in directories.items() 
        if path.exists()
    ]
    
    if not dirs_to_clean:
        print("✅ Nothing to clean. All data directories are already empty.")
        return
    
    # Show what will be deleted
    print("The following will be deleted:")
    for path, desc in dirs_to_clean:
        print(f"  - {desc} ({path})")
    
    # Ask for confirmation
    if not confirm("\nAre you sure you want to continue?"):
        print("Cleanup cancelled.")
        return
    
    # Clean directories
    print("\nCleaning up...")
    success = all(
        clean_directory(path, desc)
        for path, desc in dirs_to_clean
    )
    
    if success:
        print("\n✅ Cleanup completed successfully!")
    else:
        print("\n⚠️  Some items could not be cleaned up.")
        print("You may need to close any applications using these files.")
        sys.exit(1)

if __name__ == "__main__":
    main()
