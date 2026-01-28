#!/usr/bin/env python
"""
IDC_RIOP Application Launcher

Simple entry point to launch the IDC_RIOP GUI application.
Double-click this file or run: python launch_app.py
"""

import sys
import os
from pathlib import Path

# Determine if running as PyInstaller bundle or normal Python script
if getattr(sys, 'frozen', False):
    # Running as compiled executable - use the temp directory where PyInstaller extracts files
    BUNDLE_DIR = Path(sys._MEIPASS)
    # Working directory where user executes the .exe (for data, models, etc.)
    APPLICATION_PATH = Path(sys.executable).parent
else:
    # Running as normal Python script
    BUNDLE_DIR = Path(__file__).resolve().parent
    APPLICATION_PATH = BUNDLE_DIR

# Add project root to Python path
PROJECT_ROOT = APPLICATION_PATH
sys.path.insert(0, str(PROJECT_ROOT))

# Change to application directory so relative paths work
os.chdir(APPLICATION_PATH)

# Load environment variables from .env file (embedded in bundle or local)
def load_env_file(env_path):
    """Manually parse and load .env file without requiring python-dotenv"""
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    os.environ[key] = value
        return True
    except Exception as e:
        print(f"⚠️  Error reading .env: {e}")
        return False

# Try to load .env from bundle first (for packaged exe), then from application directory
env_loaded = False
env_path = BUNDLE_DIR / ".env"
if env_path.exists():
    if load_env_file(env_path):
        print(f"✓ Loaded environment from {env_path}")
        env_loaded = True
else:
    env_path = APPLICATION_PATH / ".env"
    if env_path.exists():
        if load_env_file(env_path):
            print(f"✓ Loaded environment from {env_path}")
            env_loaded = True

if not env_loaded:
    print(f"⚠️  .env file not found in {BUNDLE_DIR} or {APPLICATION_PATH}")

# Launch the application
if __name__ == "__main__":
    from ui.main_app import main
    main()
