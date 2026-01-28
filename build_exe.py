"""
Build Executable Script

Creates a standalone Windows executable (.exe) for the IDC_RIOP application
using PyInstaller.

Usage:
    python build_exe.py           # Build the executable
    python build_exe.py --onefile # Create single-file executable
    python build_exe.py --clean   # Clean build artifacts
"""

import subprocess
import sys
import shutil
from pathlib import Path
import argparse


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).resolve().parent


def clean_build_artifacts():
    """Remove build artifacts from previous builds."""
    project_root = get_project_root()
    
    dirs_to_remove = [
        project_root / "build",
        project_root / "dist",
    ]
    
    files_to_remove = list(project_root.glob("*.spec"))
    
    for dir_path in dirs_to_remove:
        if dir_path.exists():
            print(f"Removing: {dir_path}")
            shutil.rmtree(dir_path)
    
    for file_path in files_to_remove:
        print(f"Removing: {file_path}")
        file_path.unlink()
    
    print("Clean complete!")


def build_executable(onefile: bool = False, console: bool = False):
    """
    Build the executable using PyInstaller.
    
    Args:
        onefile: Create a single-file executable (larger but more portable)
        console: Show console window (useful for debugging)
    """
    project_root = get_project_root()
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Build the command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "IDC_RIOP",
        "--windowed" if not console else "--console",
    ]
    
    # Single file option
    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    
    # Add icon if available
    icon_path = project_root / "ui" / "assets" / "icon.ico"
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])
    
    # Add data files
    cmd.extend([
        "--add-data", f"{project_root / 'ui'};ui",
        "--add-data", f"{project_root / 'nbeats'};nbeats",
        "--add-data", f"{project_root / 'anomaly'};anomaly",
        "--add-data", f"{project_root / 'performance'};performance",
        "--add-data", f"{project_root / 'common'};common",
    ])
    
    # Hidden imports (modules that PyInstaller might miss)
    hidden_imports = [
        "pandas",
        "numpy",
        "torch",
        "scipy",
        "sklearn",
        "matplotlib",
        "plotly",
        "tqdm",
    ]
    
    for module in hidden_imports:
        cmd.extend(["--hidden-import", module])
    
    # Entry point
    cmd.append(str(project_root / "launch_app.py"))
    
    print("Building executable with command:")
    print(" ".join(cmd))
    print()
    
    # Run PyInstaller
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode == 0:
        print()
        print("=" * 60)
        print("BUILD SUCCESSFUL!")
        print("=" * 60)
        if onefile:
            exe_path = project_root / "dist" / "IDC_RIOP.exe"
        else:
            exe_path = project_root / "dist" / "IDC_RIOP" / "IDC_RIOP.exe"
        print(f"Executable location: {exe_path}")
        print()
        print("To run the application, double-click the executable or run:")
        print(f"  {exe_path}")
        print("=" * 60)
    else:
        print()
        print("BUILD FAILED!")
        print("Check the output above for errors.")
        sys.exit(1)


def create_shortcut():
    """Create a desktop shortcut for the application (Windows only)."""
    try:
        import winshell
        from win32com.client import Dispatch
    except ImportError:
        print("Installing pywin32 and winshell for shortcut creation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32", "winshell"])
        import winshell
        from win32com.client import Dispatch
    
    project_root = get_project_root()
    exe_path = project_root / "dist" / "IDC_RIOP" / "IDC_RIOP.exe"
    
    if not exe_path.exists():
        exe_path = project_root / "dist" / "IDC_RIOP.exe"
    
    if not exe_path.exists():
        print("Executable not found. Run 'python build_exe.py' first.")
        return
    
    desktop = winshell.desktop()
    shortcut_path = Path(desktop) / "IDC_RIOP.lnk"
    
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(str(shortcut_path))
    shortcut.Targetpath = str(exe_path)
    shortcut.WorkingDirectory = str(exe_path.parent)
    shortcut.Description = "IDC_RIOP Industrial Data Control Pipeline"
    
    # Set icon
    icon_path = project_root / "ui" / "assets" / "icon.ico"
    if icon_path.exists():
        shortcut.IconLocation = str(icon_path)
    
    shortcut.save()
    
    print(f"Desktop shortcut created: {shortcut_path}")


def main():
    parser = argparse.ArgumentParser(description="Build IDC_RIOP executable")
    parser.add_argument("--onefile", action="store_true", 
                       help="Create single-file executable (larger but more portable)")
    parser.add_argument("--clean", action="store_true",
                       help="Clean build artifacts")
    parser.add_argument("--console", action="store_true",
                       help="Show console window (for debugging)")
    parser.add_argument("--shortcut", action="store_true",
                       help="Create desktop shortcut (after build)")
    
    args = parser.parse_args()
    
    if args.clean:
        clean_build_artifacts()
        return
    
    if args.shortcut:
        create_shortcut()
        return
    
    build_executable(onefile=args.onefile, console=args.console)


if __name__ == "__main__":
    main()
