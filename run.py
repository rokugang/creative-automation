#!/usr/bin/env python3
"""
Main entry point for Creative Automation Platform
Run this script to start the application

Author: Rohit Gangupantulu
"""

import sys
import subprocess
import os


def main():
    """Main entry point with clear options."""
    print("\nCreative Automation Platform")
    print("=" * 40)
    print("\nAvailable options:")
    print("1. Launch Web Interface (Recommended)")
    print("2. Run Command Line Demo")
    print("3. Run Test Suite")
    print("4. Exit")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print("\nLaunching web interface...")
            print("Opening browser to http://localhost:8501")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
            
        elif choice == "2":
            print("\nRunning command line demo...")
            subprocess.run([sys.executable, "src/main.py", "demo"])
            
        elif choice == "3":
            print("\nRunning test suite...")
            subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
            
        elif choice == "4":
            print("\nExiting...")
            sys.exit(0)
            
        else:
            print("\nInvalid option. Please select 1-4.")
            main()
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    # Check if API keys are configured
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("STABILITY_API_KEY"):
        if os.path.exists(".env"):
            from dotenv import load_dotenv
            load_dotenv()
        
        if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("STABILITY_API_KEY"):
            print("\nWARNING: No API keys configured.")
            print("Please add OPENAI_API_KEY or STABILITY_API_KEY to .env file")
            print("Without API keys, image generation will not work.\n")
    
    main()
