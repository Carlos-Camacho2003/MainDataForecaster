"""
IDC_RIOP Main Application

A modern tkinter-based GUI for the IDC_RIOP industrial data pipeline.
Provides access to Training, Forecasting, and Anomaly Detection.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import sys
import os

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui.styles import COLORS, FONTS, PADDING
from ui.widgets import (
    StyledFrame, StyledLabel, StyledButton, LogConsole,
    ProgressIndicator, StatusBar
)
from ui.panels import (
    TrainingPanel, ForecastPanel,
    AnomalyPanel, AboutPanel, SharePointPanel
)

# Try to import PIL for image support
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class IDCRIOPApp:
    """Main application window for IDC_RIOP pipeline."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MainDataForecaster")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)
        
        # Store image references to prevent garbage collection
        self._images = {}
        
        # Set window icon (Logo.ico para la ventana de tkinter)
        icon_path = PROJECT_ROOT / "ui" / "assets" / "Logo.ico"
        if icon_path.exists():
            self.root.iconbitmap(str(icon_path))
        
        # Configure root window
        self.root.configure(bg=COLORS["bg_dark"])
        
        # Configure ttk styles
        self._configure_styles()
        
        # Create main layout
        self._create_layout()
        
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _configure_styles(self):
        """Configure ttk widget styles."""
        style = ttk.Style()
        style.theme_use('default')
        
        # Notebook (tabs) styling
        style.configure(
            "TNotebook",
            background=COLORS["bg_dark"],
            borderwidth=0
        )
        style.configure(
            "TNotebook.Tab",
            background=COLORS["bg_medium"],
            foreground=COLORS["text_secondary"],
            padding=[15, 8],
            font=FONTS["body_bold"]
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", COLORS["bg_light"])],
            foreground=[("selected", COLORS["text_primary"])]
        )
        
        # Separator
        style.configure(
            "TSeparator",
            background=COLORS["bg_light"]
        )
        
        # Combobox
        style.configure(
            "TCombobox",
            fieldbackground=COLORS["input_bg"],
            background=COLORS["bg_medium"],
            foreground=COLORS["text_primary"],
            arrowcolor=COLORS["text_primary"]
        )
    
    def _create_layout(self):
        """Create the main application layout."""
        # Main container
        main_container = StyledFrame(self.root, style="dark")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self._create_header(main_container)
        
        # Content area (notebook + log)
        content_frame = StyledFrame(main_container, style="dark")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=PADDING["medium"], pady=PADDING["small"])
        
        # Right side: Log console (create FIRST so panels can reference it)
        log_frame = StyledFrame(content_frame, style="dark")
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(PADDING["medium"], 0))
        
        StyledLabel(log_frame, text="Registro de Salida", style="subheading").pack(anchor=tk.W)
        
        self.log_console = LogConsole(log_frame, height=30, width=50)
        self.log_console.pack(fill=tk.BOTH, expand=True, pady=(PADDING["small"], 0))
        
        # Clear log button
        clear_btn = StyledButton(
            log_frame,
            text="Limpiar Registro",
            command=self.log_console.clear,
            style="secondary",
            width=14
        )
        clear_btn.pack(anchor=tk.E, pady=(PADDING["small"], 0))
        
        # Status bar (create BEFORE panels so they can reference it)
        self.status_bar = StatusBar(main_container)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Left side: Notebook with operation panels
        notebook_frame = StyledFrame(content_frame, style="dark")
        notebook_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create panels (AFTER log_console and status_bar are created)
        self._create_panels()
        
        # Initial log message
        self.log_console.log("Pipeline IDC_RIOP inicializado", "success")
        self.log_console.log(f"Directorio raíz: {PROJECT_ROOT}", "info")
        self.log_console.log("Seleccione una operación de las pestañas para comenzar.", "info")
    
    def _create_header(self, parent):
        """Create the application header."""
        header_frame = StyledFrame(parent, style="medium")
        header_frame.pack(fill=tk.X)
        
        # Inner container for padding
        header_inner = StyledFrame(header_frame, style="medium")
        header_inner.pack(fill=tk.X, padx=PADDING["large"], pady=PADDING["medium"])
        
        # Logo and title
        title_frame = StyledFrame(header_inner, style="medium")
        title_frame.pack(side=tk.LEFT)
        
        # Load logo image
        logo_path = PROJECT_ROOT / "ui" / "assets" / "riopaila_logo.png"
        if logo_path.exists() and HAS_PIL:
            try:
                # Load and resize logo
                img = Image.open(logo_path)
                # Resize to fit header (height ~50px)
                aspect_ratio = img.width / img.height
                new_height = 50
                new_width = int(new_height * aspect_ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                self._images["logo"] = ImageTk.PhotoImage(img)
                
                logo_label = tk.Label(
                    title_frame,
                    image=self._images["logo"],
                    bg=COLORS["bg_medium"]
                )
                logo_label.pack(side=tk.LEFT, padx=(0, PADDING["medium"]))
            except Exception as e:
                # Fallback to text if image loading fails
                tk.Label(
                    title_frame,
                    text="🏭",
                    font=("Segoe UI Emoji", 28),
                    bg=COLORS["bg_medium"],
                    fg=COLORS["accent"]
                ).pack(side=tk.LEFT, padx=(0, PADDING["medium"]))
        else:
            # Fallback to emoji if no PIL or image not found
            tk.Label(
                title_frame,
                text="🏭",
                font=("Segoe UI Emoji", 28),
                bg=COLORS["bg_medium"],
                fg=COLORS["accent"]
            ).pack(side=tk.LEFT, padx=(0, PADDING["medium"]))
        
        text_frame = StyledFrame(title_frame, style="medium")
        text_frame.pack(side=tk.LEFT)
        
        tk.Label(
            text_frame,
            text="MainDataForecaster",
            font=FONTS["title"],
            bg=COLORS["bg_medium"],
            fg=COLORS["text_primary"]
        ).pack(anchor=tk.W)
        
        tk.Label(
            text_frame,
            text="Predicción de desempeño y detección de anomalías de Desfibradora y Picadora",
            font=FONTS["small"],
            bg=COLORS["bg_medium"],
            fg=COLORS["text_secondary"]
        ).pack(anchor=tk.W)
        
        # Quick action buttons on right
        actions_frame = StyledFrame(header_inner, style="medium")
        actions_frame.pack(side=tk.RIGHT)
        
        StyledButton(
            actions_frame,
            text="📂 Abrir Datos",
            command=self._open_data_folder,
            style="secondary",
            width=14
        ).pack(side=tk.LEFT, padx=PADDING["small"])
        
        StyledButton(
            actions_frame,
            text="📊 Ver Resultados",
            command=self._open_results_folder,
            style="secondary",
            width=14
        ).pack(side=tk.LEFT, padx=PADDING["small"])
    
    def _create_panels(self):
        """Create the operation panels in the notebook."""
        # SharePoint Panel
        sharepoint_frame = StyledFrame(self.notebook, style="dark")
        sharepoint_panel = SharePointPanel(
            sharepoint_frame,
            log_callback=self.log_console.log,
            status_callback=self.status_bar.set_status
        )
        sharepoint_panel.pack(fill=tk.BOTH, expand=True, padx=PADDING["large"], pady=PADDING["large"])
        self.notebook.add(sharepoint_frame, text="  🔄 SharePoint  ")

        # Training Panel
        train_frame = StyledFrame(self.notebook, style="dark")
        train_panel = TrainingPanel(
            train_frame,
            log_callback=self.log_console.log,
            status_callback=self.status_bar.set_status
        )
        train_panel.pack(fill=tk.BOTH, expand=True, padx=PADDING["large"], pady=PADDING["large"])
        self.notebook.add(train_frame, text="  🧠 Entrenamiento  ")
        
        # Forecast Panel
        forecast_frame = StyledFrame(self.notebook, style="dark")
        forecast_panel = ForecastPanel(
            forecast_frame,
            log_callback=self.log_console.log,
            status_callback=self.status_bar.set_status
        )
        forecast_panel.pack(fill=tk.BOTH, expand=True, padx=PADDING["large"], pady=PADDING["large"])
        self.notebook.add(forecast_frame, text="  📈 Pronósticos  ")
        
        # Anomaly Panel
        anomaly_frame = StyledFrame(self.notebook, style="dark")
        anomaly_panel = AnomalyPanel(
            anomaly_frame,
            log_callback=self.log_console.log,
            status_callback=self.status_bar.set_status
        )
        anomaly_panel.pack(fill=tk.BOTH, expand=True, padx=PADDING["large"], pady=PADDING["large"])
        self.notebook.add(anomaly_frame, text="  🔍 Detección de Anomalías  ")
        
        # About Panel
        about_frame = StyledFrame(self.notebook, style="dark")
        about_panel = AboutPanel(about_frame)
        about_panel.pack(fill=tk.BOTH, expand=True, padx=PADDING["large"], pady=PADDING["large"])
        self.notebook.add(about_frame, text="  ℹ Acerca de  ")
    
    def _open_data_folder(self):
        """Open the data folder in file explorer."""
        data_path = PROJECT_ROOT / "data"
        if data_path.exists():
            os.startfile(str(data_path))
        else:
            messagebox.showwarning("Advertencia", f"Carpeta de datos no encontrada: {data_path}")
    
    def _open_results_folder(self):
        """Open the performance results folder in file explorer."""
        results_path = PROJECT_ROOT / "performance_results"
        if results_path.exists():
            os.startfile(str(results_path))
        else:
            # Try forecasts folder as fallback
            forecasts_path = PROJECT_ROOT / "forecasts"
            if forecasts_path.exists():
                os.startfile(str(forecasts_path))
            else:
                messagebox.showwarning("Advertencia", "Carpeta de resultados no encontrada. Ejecute un pipeline primero.")
    
    def _on_close(self):
        """Handle window close event."""
        # Check if any operation is running
        # For now, just close
        self.root.destroy()
    
    def run(self):
        """Start the application main loop."""
        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Start main loop
        self.root.mainloop()


def main():
    """Entry point for the application."""
    app = IDCRIOPApp()
    app.run()


if __name__ == "__main__":
    main()
