# IDC_RIOP User Interface

A modern graphical user interface for the IDC_RIOP Industrial Data Control Pipeline.

## Features

- **Performance Pipeline** - Calculate Equipment Performance Index (EPI) and Plant Performance Index (PPI)
- **N-BEATS Training** - Train deep learning models for time series forecasting
- **N-BEATS Forecasting** - Generate multi-horizon forecasts with uncertainty quantification
- **Anomaly Detection** - Detect and clean sensor data anomalies

## Quick Start

### Running the Application

**Option 1: Direct Python Launch**
```bash
python launch_app.py
```

**Option 2: From Module**
```python
from ui.main_app import main
main()
```

### Building an Executable

To create a standalone Windows executable (.exe):

```bash
# Standard build (folder with executable)
python build_exe.py

# Single-file executable (larger but more portable)
python build_exe.py --onefile

# With console window (for debugging)
python build_exe.py --console

# Clean build artifacts
python build_exe.py --clean
```

The executable will be created in the `dist/` folder.

### Creating a Desktop Shortcut

After building the executable:
```bash
python build_exe.py --shortcut
```

## Project Structure

```
ui/
├── __init__.py          # Package initialization
├── main_app.py          # Main application window
├── panels.py            # Operation panels (Performance, Training, etc.)
├── widgets.py           # Custom styled widgets
├── styles.py            # Color palette, fonts, styling constants
└── assets/              # Icons and other assets
    └── icon.ico         # Application icon (optional)
```

## Dependencies

The UI uses only tkinter (included with Python) for the graphical interface. For building executables:

```
pyinstaller>=6.0.0
pywin32>=306  # For Windows shortcuts
winshell>=0.6  # For desktop shortcuts
```

## Customization

### Changing Colors

Edit `ui/styles.py` to modify the color palette:

```python
COLORS = {
    "bg_dark": "#1a1a2e",      # Main background
    "accent": "#e94560",        # Accent color (buttons, highlights)
    ...
}
```

### Adding New Panels

1. Create a new panel class in `ui/panels.py` inheriting from `BasePanel`
2. Add the panel to `_create_panels()` in `ui/main_app.py`

## Screenshots

The application features a modern dark theme with:
- Tabbed interface for different operations
- Real-time log output
- Progress indicators
- Configuration options for each pipeline

## Troubleshooting

### Application doesn't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### Executable build fails
- Install PyInstaller: `pip install pyinstaller`
- Run with `--console` flag to see error messages

### Operations fail
- Check the log console for error messages
- Ensure data files exist in the `data/` folder
- Verify Python environment has required packages (torch, pandas, etc.)
