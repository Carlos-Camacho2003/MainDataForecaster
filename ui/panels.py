"""
Operation Panels for IDC_RIOP UI

Each panel provides controls for a specific operation (Performance, Training, etc.)
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Callable, Optional
import threading
import pandas as pd
import shutil
import os
import webbrowser

from ui.styles import COLORS, FONTS, PADDING
from ui.widgets import (
    StyledButton, StyledLabel, StyledFrame, StyledLabelFrame,
    StyledCombobox, StyledCheckbutton, StyledSpinbox, LogConsole, ProgressIndicator,
    ScrollableFrame
)


class BasePanel(StyledFrame):
    """Base class for operation panels."""
    
    def __init__(self, parent, log_callback: Callable, status_callback: Callable, **kwargs):
        super().__init__(parent, **kwargs)
        self.log = log_callback
        self.set_status = status_callback
        self._running = False
    
    def is_running(self) -> bool:
        return self._running
    
    def set_running(self, running: bool):
        self._running = running


class TrainingPanel(BasePanel):
    """Panel for N-BEATS Training operations."""
    
    def __init__(self, parent, log_callback: Callable, status_callback: Callable, **kwargs):
        super().__init__(parent, log_callback, status_callback, **kwargs)
        self._setup_ui()
    
    def _setup_ui(self):
        # Create scrollable container for all content
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill=tk.BOTH, expand=True)
        
        # Use scrollable.interior as the content frame
        content_frame = scrollable.interior
        
        # Title
        StyledLabel(
            content_frame, 
            text="🧠 Entrenamiento N-BEATS",
            style="heading"
        ).pack(anchor=tk.W, pady=(0, PADDING["medium"]))
        
        # Description
        StyledLabel(
            content_frame,
            text="Entrenar modelos de aprendizaje profundo para pronóstico de series de tiempo.\n"
                 "Los modelos se entrenan con datos de sensores agregados por hora.",
            style="body"
        ).pack(anchor=tk.W, pady=(0, PADDING["large"]))
        
        # Machine selection frame
        machine_frame = StyledLabelFrame(content_frame, text="Selección de Máquina")
        machine_frame.pack(fill=tk.X, pady=PADDING["medium"])
        
        machine_inner = StyledFrame(machine_frame)
        machine_inner.pack(fill=tk.X, pady=PADDING["small"])
        
        StyledLabel(machine_inner, text="Máquina:", style="body_bold").pack(side=tk.LEFT)
        self.machine_combo = StyledCombobox(
            machine_inner,
            values=["TODO", "DESF", "PICADORA", "PLANT"]
        )
        self.machine_combo.pack(side=tk.LEFT, padx=(PADDING["medium"], 0))
        
        # Model configuration frame
        config_frame = StyledLabelFrame(content_frame, text="Configuración del Modelo")
        config_frame.pack(fill=tk.X, pady=PADDING["medium"])
        
        # Model size
        size_frame = StyledFrame(config_frame)
        size_frame.pack(fill=tk.X, pady=PADDING["small"])
        
        StyledLabel(size_frame, text="Tamaño del Modelo:", style="body_bold").pack(side=tk.LEFT)
        self.size_combo = StyledCombobox(
            size_frame,
            values=["pequeño (~100K parámetros)", "mediano (~1.5M parámetros)", "grande (~2M parámetros)"],
            width=28
        )
        self.size_combo.current(1)  # Default to medium
        self.size_combo.pack(side=tk.LEFT, padx=(PADDING["medium"], 0))
        
        # Epochs
        epochs_frame = StyledFrame(config_frame)
        epochs_frame.pack(fill=tk.X, pady=PADDING["small"])
        
        StyledLabel(epochs_frame, text="Épocas:", style="body_bold").pack(side=tk.LEFT)
        self.epochs_spin = StyledSpinbox(epochs_frame, from_=10, to=500, value=100)
        self.epochs_spin.pack(side=tk.LEFT, padx=(PADDING["medium"], 0))
        
        # Aggregation method
        agg_frame = StyledFrame(config_frame)
        agg_frame.pack(fill=tk.X, pady=PADDING["small"])
        
        StyledLabel(agg_frame, text="Agregación:", style="body_bold").pack(side=tk.LEFT)
        self.agg_combo = StyledCombobox(
            agg_frame,
            values=["media (más suave, tendencias)", "máximo (picos, alarmas)"],
            width=28
        )
        self.agg_combo.pack(side=tk.LEFT, padx=(PADDING["medium"], 0))
        # Default to 'máximo' (index 1) so UI uses peak aggregation for alarms
        self.agg_combo.current(1)
        
        # Seed for reproducibility
        seed_frame = StyledFrame(config_frame)
        seed_frame.pack(fill=tk.X, pady=PADDING["small"])
        
        StyledLabel(seed_frame, text="Semilla (seed):", style="body_bold").pack(side=tk.LEFT)
        self.seed_entry = tk.Entry(
            seed_frame,
            font=FONTS["body"],
            bg=COLORS["input_bg"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"],
            width=12
        )
        self.seed_entry.pack(side=tk.LEFT, padx=(PADDING["medium"], 0))
        StyledLabel(seed_frame, text="(opcional, para reproducibilidad)", style="small").pack(side=tk.LEFT, padx=(PADDING["small"], 0))
        
        # Options frame
        options_frame = StyledLabelFrame(content_frame, text="Opciones de Entrenamiento")
        options_frame.pack(fill=tk.X, pady=PADDING["medium"])
        
        self.force_var = tk.BooleanVar(value=False)
        StyledCheckbutton(
            options_frame,
            text="Forzar reentrenamiento de modelos existentes",
            variable=self.force_var
        ).pack(anchor=tk.W, pady=PADDING["small"])
        
        self.include_epi_var = tk.BooleanVar(value=True)
        StyledCheckbutton(
            options_frame,
            text="Incluir entrenamiento de EPI/PPI",
            variable=self.include_epi_var
        ).pack(anchor=tk.W, pady=PADDING["small"])
        
        self.clean_anomalies_var = tk.BooleanVar(value=True)
        StyledCheckbutton(
            options_frame,
            text="Limpiar anomalías antes de entrenar (recomendado)",
            variable=self.clean_anomalies_var
        ).pack(anchor=tk.W, pady=PADDING["small"])
        
        # Button frame
        btn_frame = StyledFrame(content_frame)
        btn_frame.pack(fill=tk.X, pady=PADDING["large"])
        
        # Run button
        self.run_btn = StyledButton(
            btn_frame,
            text="▶ Iniciar Entrenamiento",
            command=self._run_training,
            style="primary",
            width=25
        )
        self.run_btn.pack(anchor=tk.CENTER)
        
        # Warning
        StyledLabel(
            content_frame,
            text="⚠ El entrenamiento puede tomar varios minutos u horas dependiendo del tamaño de datos y configuración.",
            style="small",
            fg=COLORS["warning"]
        ).pack(anchor=tk.CENTER)
        
        # Copyright
        ttk.Separator(content_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=PADDING["large"]
        )
        StyledLabel(
            content_frame,
            text="© IDC Ingeniería de Confiabilidad | Riopaila Castilla",
            style="small",
            fg=COLORS["text_muted"]
        ).pack(anchor=tk.CENTER, pady=PADDING["small"])
    
    def _run_training(self):
        if self._running:
            return
        
        self.set_running(True)
        self.run_btn.set_enabled(False)
        self.set_status("Entrenando modelos N-BEATS...", "info")
        
        def run():
            try:
                self.log("=" * 60, "heading")
                self.log("Iniciando Entrenamiento N-BEATS", "heading")
                self.log("=" * 60, "heading")
                
                import sys
                import os
                from pathlib import Path
                
                project_root = Path(__file__).resolve().parent.parent
                sys.path.insert(0, str(project_root))
                
                # Change to project directory so relative paths work
                os.chdir(project_root)
                
                from nbeats.nbeats_train import batch_train_machine, batch_train_all
                
                machine = self.machine_combo.get()
                if machine == "TODO":
                    machine = "ALL"
                    
                size_map = {
                    "pequeño (~100K parámetros)": "small",
                    "mediano (~1.5M parámetros)": "medium",
                    "grande (~2M parámetros)": "large"
                }
                model_size = size_map.get(self.size_combo.get(), "medium")
                
                agg_map = {
                    "media (más suave, tendencias)": "mean",
                    "máximo (picos, alarmas)": "max"
                }
                aggregation = agg_map.get(self.agg_combo.get(), "mean")
                
                epochs = int(self.epochs_spin.get())
                
                # Parse seed (optional)
                seed_text = self.seed_entry.get().strip()
                seed = int(seed_text) if seed_text else None
                
                self.log(f"Configuración:", "info")
                self.log(f"  Máquina: {machine}", "info")
                self.log(f"  Tamaño del modelo: {model_size}", "info")
                self.log(f"  Épocas: {epochs}", "info")
                self.log(f"  Agregación: {aggregation}", "info")
                self.log(f"  Semilla: {seed if seed is not None else 'aleatoria'}", "info")
                self.log(f"  Forzar reentrenamiento: {self.force_var.get()}", "info")
                self.log(f"  Incluir EPI: {self.include_epi_var.get()}", "info")
                self.log(f"  Limpiar anomalías: {self.clean_anomalies_var.get()}", "info")
                self.log("", "info")
                
                kwargs = {
                    "include_epi": self.include_epi_var.get(),
                    "epochs": epochs,
                    "model_size": model_size,
                    "skip_existing": not self.force_var.get(),
                    "aggregation": aggregation,
                    "clean_anomalies": self.clean_anomalies_var.get(),
                    "seed": seed
                }
                
                if machine == "ALL":
                    results = batch_train_all(**kwargs)
                else:
                    results = batch_train_machine(machine, **kwargs)
                
                self.log("=" * 60, "heading")
                self.log("✅ ¡Entrenamiento completado!", "success")
                self.log("=" * 60, "heading")
                self.set_status("Entrenamiento completado", "success")
                
            except Exception as e:
                self.log(f"❌ Error: {str(e)}", "error")
                import traceback
                self.log(traceback.format_exc(), "error")
                self.set_status(f"Error: {str(e)}", "error")
            finally:
                self.set_running(False)
                self.run_btn.set_enabled(True)
        
        threading.Thread(target=run, daemon=True).start()


class ForecastPanel(BasePanel):
    """Panel for N-BEATS Forecasting operations."""
    
    def __init__(self, parent, log_callback: Callable, status_callback: Callable, **kwargs):
        super().__init__(parent, log_callback, status_callback, **kwargs)
        self._calculate_dynamic_horizons()
        self._setup_ui()
    
    def _calculate_dynamic_horizons(self):
        """Calculate horizon options based on available processed data."""
        self.horizon_config = {}
        
        # Default fallback
        default_options = [
            "2_days (48h, ~85% precisión, ALTA confiabilidad)",
            "5_days (120h, ~72% precisión, MEDIA confiabilidad)",
            "15_days (360h, ~55% precisión, BAJA confiabilidad)",
            "1_month (720h, ~42% precisión, solo TENDENCIA)"
        ]
        self.horizon_config["default"] = default_options
        
        def scan_data():
            try:
                # Use absolute path relative to this file
                project_root = Path(__file__).resolve().parent.parent
                processed_dir = project_root / "processed"
                
                if not processed_dir.exists():
                    print(f"Processed dir not found at {processed_dir}")
                    return

                global_max_samples = 0
                global_freq = 1.0

                for machine in ["DESF", "PICADORA", "PLANT"]:
                    machine_dir = processed_dir / machine
                    if machine_dir.exists():
                        max_samples = 0
                        freq_hours = 1.0
                        
                        # Scan parquet files
                        for f in machine_dir.glob("*.parquet"):
                            try:
                                # Read just the shape first if possible, or just index
                                # Parquet metadata read is fast
                                df = pd.read_parquet(f, columns=['timestamp'])
                                if len(df) > max_samples:
                                    max_samples = len(df)
                                    if len(df) > 1:
                                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                                        freq_hours = df['timestamp'].diff().median().total_seconds() / 3600
                            except Exception:
                                continue
                        
                        if max_samples > 0:
                            self.horizon_config[machine] = self._generate_options_for_samples(max_samples, freq_hours)
                            if max_samples > global_max_samples:
                                global_max_samples = max_samples
                                global_freq = freq_hours
                
                # Set default/TODO options based on the best available data
                if global_max_samples > 0:
                    best_options = self._generate_options_for_samples(global_max_samples, global_freq)
                    self.horizon_config["default"] = best_options
                    self.horizon_config["TODO"] = best_options
                            
                # Update UI if currently showing a machine
                self.after(0, self._update_horizon_options)
                
            except Exception as e:
                print(f"Error calculating horizons: {e}")

        # Run in thread to avoid blocking UI startup
        threading.Thread(target=scan_data, daemon=True).start()

    def _generate_options_for_samples(self, n_samples, freq_hours=1.0):
        # Expanded intervals to support larger datasets
        intervals = [
            ("2_days", 48),
            ("5_days", 120),
            ("7_days", 168),
            ("15_days", 360),
            ("1_month", 720),
            ("2_months", 1440),
            ("3_months", 2160)
        ]
        
        options = []
        for label, hours in intervals:
            steps = int(hours / freq_hours)
            ratio = steps / n_samples if n_samples > 0 else 1.0
            
            # Skip if horizon is > 50% of history (too unreliable)
            if ratio > 0.5:
                continue
            
            if ratio <= 0.10:
                precision = "~85-90%"
                conf = "ALTA confiabilidad"
            elif ratio <= 0.15:
                precision = "~75-85%"
                conf = "ALTA-MEDIA confiabilidad"
            elif ratio <= 0.25:
                precision = "~65-75%"
                conf = "MEDIA confiabilidad"
            elif ratio <= 0.40:
                precision = "~50-65%"
                conf = "BAJA confiabilidad"
            else:
                precision = "<50%"
                conf = "solo TENDENCIA"
                
            options.append(f"{label} ({hours}h, {precision} precisión, {conf})")
        return options

    def _update_horizon_options(self, event=None):
        """Update horizon options based on selected machine."""
        machine = self.machine_combo.get()
        
        # Default fallback options (updated to match dynamic style)
        options = [
            "2_days (48h, ~85-90% precisión, ALTA confiabilidad)",
            "5_days (120h, ~75-85% precisión, ALTA-MEDIA confiabilidad)",
            "7_days (168h, ~65-75% precisión, MEDIA confiabilidad)",
            "15_days (360h, ~50-65% precisión, BAJA confiabilidad)",
            "1_month (720h, <50% precisión, solo TENDENCIA)"
        ]
        
        # Try to find dynamic options
        if hasattr(self, 'horizon_config') and self.horizon_config:
            if machine in self.horizon_config:
                options = self.horizon_config[machine]
            elif "default" in self.horizon_config:
                options = self.horizon_config["default"]
        
        self.horizon_combo['values'] = options
        if options:
            self.horizon_combo.current(0)

    def _visualize_horizon_selection(self):
        """Visualize the selected horizon against available data."""
        machine = self.machine_combo.get()
        horizon_str = self.horizon_combo.get()
        
        if not machine or not horizon_str:
            return
            
        # Parse horizon hours
        try:
            # Format: "2_days (48h, ...)"
            import re
            match = re.search(r'\((\d+)h', horizon_str)
            if not match:
                return
            horizon_hours = int(match.group(1))
        except Exception:
            return

        # Get data info
        project_root = Path(__file__).resolve().parent.parent
        processed_dir = project_root / "processed"
        
        machines_to_check = [machine] if machine != "TODO" else ["DESF", "PICADORA", "PLANT"]
        
        data_info = []
        
        for m in machines_to_check:
            machine_dir = processed_dir / m
            if machine_dir.exists():
                for f in machine_dir.glob("*.parquet"):
                    try:
                        df = pd.read_parquet(f, columns=['timestamp'])
                        if not df.empty:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            start_date = df['timestamp'].min()
                            end_date = df['timestamp'].max()
                            duration_hours = (end_date - start_date).total_seconds() / 3600
                            data_info.append({
                                'machine': m,
                                'start': start_date,
                                'end': end_date,
                                'duration_hours': duration_hours,
                                'samples': len(df)
                            })
                    except Exception:
                        pass
        
        if not data_info:
            messagebox.showinfo("Info", "No se encontraron datos para visualizar.")
            return
            
        # Generate Plotly Visualization
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            for i, info in enumerate(data_info):
                # Historical Data Bar
                fig.add_trace(go.Bar(
                    x=[info['duration_hours']],
                    y=[info['machine']],
                    orientation='h',
                    name='Histórico Disponible',
                    marker=dict(color='blue', opacity=0.6),
                    text=f"{info['duration_hours']:.1f}h ({info['samples']} muestras)",
                    textposition='auto'
                ))
                
                # Horizon Bar (overlay)
                fig.add_trace(go.Bar(
                    x=[horizon_hours],
                    y=[info['machine']],
                    orientation='h',
                    name='Horizonte Pronóstico',
                    marker=dict(color='red', opacity=0.6),
                    text=f"{horizon_hours}h",
                    textposition='inside'
                ))
            
            fig.update_layout(
                title=f"Cobertura de Datos vs Horizonte ({horizon_hours}h)",
                xaxis_title="Horas",
                yaxis_title="Máquina",
                barmode='overlay',
                template="plotly_white"
            )
            
            # Save to temp file and open
            temp_file = project_root / "temp_horizon_viz.html"
            fig.write_html(str(temp_file))
            webbrowser.open(f"file://{temp_file}")
            
        except ImportError:
            messagebox.showerror("Error", "Plotly no está instalado.")
        except Exception as e:
            messagebox.showerror("Error", f"Error generando visualización: {e}")

    def _setup_ui(self):
        # Create scrollable container for all content
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill=tk.BOTH, expand=True)
        
        # Use scrollable.interior as the content frame
        content_frame = scrollable.interior
        
        # Title
        StyledLabel(
            content_frame, 
            text="📈 Pronósticos N-BEATS",
            style="heading"
        ).pack(anchor=tk.W, pady=(0, PADDING["medium"]))
        
        # Description
        StyledLabel(
            content_frame,
            text="Generar pronósticos con cuantificación de incertidumbre.\n"
                 "Soporta múltiples horizontes de pronóstico con estimaciones de precisión.",
            style="body"
        ).pack(anchor=tk.W, pady=(0, PADDING["large"]))
        
        # Machine selection
        machine_frame = StyledLabelFrame(content_frame, text="Selección de Máquina")
        machine_frame.pack(fill=tk.X, pady=PADDING["medium"])
        
        machine_inner = StyledFrame(machine_frame)
        machine_inner.pack(fill=tk.X, pady=PADDING["small"])
        
        StyledLabel(machine_inner, text="Máquina:", style="body_bold").pack(side=tk.LEFT)
        self.machine_combo = StyledCombobox(
            machine_inner,
            values=["TODO", "DESF", "PICADORA", "PLANT"]
        )
        self.machine_combo.pack(side=tk.LEFT, padx=(PADDING["medium"], 0))
        self.machine_combo.bind("<<ComboboxSelected>>", self._update_horizon_options)
        
        # Horizon selection
        horizon_frame = StyledLabelFrame(content_frame, text="Horizonte de Pronóstico")
        horizon_frame.pack(fill=tk.X, pady=PADDING["medium"])
        
        horizon_inner = StyledFrame(horizon_frame)
        horizon_inner.pack(fill=tk.X, pady=PADDING["small"])
        
        StyledLabel(horizon_inner, text="Horizonte:", style="body_bold").pack(side=tk.LEFT)
        self.horizon_combo = StyledCombobox(
            horizon_inner,
            values=[],
            width=48
        )
        self.horizon_combo.pack(side=tk.LEFT, padx=(PADDING["medium"], 0))
        
        # Add Visualize Button
        StyledButton(
            horizon_inner,
            text="👁 Ver",
            command=self._visualize_horizon_selection,
            style="secondary",
            width=8
        ).pack(side=tk.LEFT, padx=PADDING["small"])
        
        # Initialize options
        self._update_horizon_options()
        
        # Options
        options_frame = StyledLabelFrame(content_frame, text="Opciones")
        options_frame.pack(fill=tk.X, pady=PADDING["medium"])
        
        # Aggregation selection (matches training terminology)
        agg_frame = StyledFrame(options_frame)
        agg_frame.pack(fill=tk.X, pady=PADDING["small"])
        StyledLabel(agg_frame, text="Agregación (sensores):", style="body_bold").pack(side=tk.LEFT)
        self.agg_combo = StyledCombobox(
            agg_frame,
            values=[
                "media (más suave, tendencias)",
                "máximo (picos, alarmas)"
            ],
            width=32
        )
        self.agg_combo.pack(side=tk.LEFT, padx=(PADDING["medium"], 0))
        # Default to 'máximo' (index 1) to align with pipeline's peak-based alerts
        self.agg_combo.current(1)

        # MC samples
        mc_frame = StyledFrame(options_frame)
        mc_frame.pack(fill=tk.X, pady=PADDING["small"])
        
        StyledLabel(mc_frame, text="Muestras Monte Carlo:", style="body_bold").pack(side=tk.LEFT)
        self.mc_spin = StyledSpinbox(mc_frame, from_=10, to=500, value=100)
        self.mc_spin.pack(side=tk.LEFT, padx=(PADDING["medium"], 0))
        
        self.clean_anomalies_var = tk.BooleanVar(value=True)
        StyledCheckbutton(
            options_frame,
            text="Limpiar anomalías antes de pronosticar",
            variable=self.clean_anomalies_var
        ).pack(anchor=tk.W, pady=PADDING["small"])
        
        # Seed for reproducibility
        seed_frame = StyledFrame(options_frame)
        seed_frame.pack(fill=tk.X, pady=PADDING["small"])
        
        StyledLabel(seed_frame, text="Semilla (seed):", style="body_bold").pack(side=tk.LEFT)
        self.seed_entry = tk.Entry(
            seed_frame,
            font=FONTS["body"],
            bg=COLORS["input_bg"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"],
            width=12
        )
        self.seed_entry.pack(side=tk.LEFT, padx=(PADDING["medium"], 0))
        StyledLabel(seed_frame, text="(opcional, para reproducibilidad)", style="small").pack(side=tk.LEFT, padx=(PADDING["small"], 0))
        
        # Button frame
        btn_frame = StyledFrame(content_frame)
        btn_frame.pack(fill=tk.X, pady=PADDING["large"])
        
        # Run button
        self.run_btn = StyledButton(
            btn_frame,
            text="▶ Generar Pronósticos",
            command=self._run_forecast,
            style="primary",
            width=25
        )
        self.run_btn.pack(anchor=tk.CENTER)
        
        # Output info
        StyledLabel(
            content_frame,
            text="Ubicaciones de salida:\n"
                 "  • Visualizaciones individuales: forecast_visuals/{MACHINE}/\n"
                 "  • CSVs combinados: data/forecasts/{MACHINE}/FORECAST_{MACHINE}_{HORIZON}.csv",
            style="small"
        ).pack(anchor=tk.CENTER, pady=PADDING["medium"])
        
        # Copyright
        ttk.Separator(content_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=PADDING["large"]
        )
        StyledLabel(
            content_frame,
            text="© IDC Ingeniería de Confiabilidad | Riopaila Castilla",
            style="small",
            fg=COLORS["text_muted"]
        ).pack(anchor=tk.CENTER, pady=PADDING["small"])
    
    def _run_forecast(self):
        if self._running:
            return
        
        self.set_running(True)
        self.run_btn.set_enabled(False)
        self.set_status("Generando pronósticos...", "info")
        
        def run():
            try:
                self.log("=" * 60, "heading")
                self.log("Iniciando Pronósticos N-BEATS", "heading")
                self.log("=" * 60, "heading")
                
                import sys
                import os
                from pathlib import Path
                
                project_root = Path(__file__).resolve().parent.parent
                sys.path.insert(0, str(project_root))
                os.chdir(project_root)
                
                from nbeats.nbeats_forecast import batch_forecast_to_data
                from nbeats.nbeats_train import prepare_all_data_for_machine
                
                machine = self.machine_combo.get()
                if machine == "TODO":
                    machine = "ALL"
                
                # Extract horizon from combo
                horizon_text = self.horizon_combo.get()
                horizon = horizon_text.split(" ")[0]  # e.g., "2_days"
                
                mc_samples = int(self.mc_spin.get())

                agg_map = {
                    "media (más suave, tendencias)": "mean",
                    "máximo (picos, alarmas)": "max"
                }
                aggregation = agg_map.get(self.agg_combo.get(), "mean")
                
                # Parse seed (optional)
                seed_text = self.seed_entry.get().strip()
                seed = int(seed_text) if seed_text else None
                
                self.log(f"Configuración:", "info")
                self.log(f"  Máquina: {machine}", "info")
                self.log(f"  Horizonte: {horizon}", "info")
                self.log(f"  Muestras MC: {mc_samples}", "info")
                self.log(f"  Agregación (sensores): {aggregation}", "info")
                self.log(f"  Semilla: {seed if seed is not None else 'aleatoria'}", "info")
                self.log(f"  Limpiar anomalías: {self.clean_anomalies_var.get()}", "info")
                self.log("", "info")
                
                # PHASE 0: Data Preparation
                self.log("Preparando datos más recientes...", "info")
                
                machines_to_prep = []
                if machine == "ALL":
                    machines_to_prep = ["DESF", "PICADORA"]
                elif machine in ["DESF", "PICADORA"]:
                    machines_to_prep = [machine]
                
                for m in machines_to_prep:
                    self.log(f"  Procesando datos para {m}...", "info")
                    try:
                        prepare_all_data_for_machine(m, aggregation=aggregation)
                    except Exception as e:
                        self.log(f"  ⚠️ Error preparando datos para {m}: {e}", "warning")
                
                batch_forecast_to_data(
                    horizon_preset=horizon,
                    machine=machine if machine != "ALL" else None,
                    n_mc_samples=mc_samples,
                    clean_anomalies=self.clean_anomalies_var.get(),
                    seed=seed
                )
                
                self.log("=" * 60, "heading")
                self.log("✅ ¡Pronósticos completados!", "success")
                self.log("=" * 60, "heading")
                self.set_status("Pronósticos completados", "success")
                
            except Exception as e:
                self.log(f"❌ Error: {str(e)}", "error")
                import traceback
                self.log(traceback.format_exc(), "error")
                self.set_status(f"Error: {str(e)}", "error")
            finally:
                self.set_running(False)
                self.run_btn.set_enabled(True)
        
        threading.Thread(target=run, daemon=True).start()


class AnomalyPanel(BasePanel):
    """Panel for Anomaly Detection operations."""
    
    def __init__(self, parent, log_callback: Callable, status_callback: Callable, **kwargs):
        super().__init__(parent, log_callback, status_callback, **kwargs)
        self._setup_ui()
    
    def _setup_ui(self):
        # Create scrollable container for all content
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill=tk.BOTH, expand=True)
        
        # Use scrollable.interior as the content frame
        content_frame = scrollable.interior
        
        # Title
        StyledLabel(
            content_frame, 
            text="🔍 Detección de Anomalías",
            style="heading"
        ).pack(anchor=tk.W, pady=(0, PADDING["medium"]))
        
        # Description
        StyledLabel(
            content_frame,
            text="Detectar anomalías en datos de sensores.\n"
                 "Utiliza múltiples métodos de detección con ensamble por votación.",
            style="body"
        ).pack(anchor=tk.W, pady=(0, PADDING["large"]))
        
        # Machine selection
        machine_frame = StyledFrame(content_frame)
        machine_frame.pack(fill=tk.X, pady=PADDING["medium"])
        
        StyledLabel(machine_frame, text="Máquina:", style="body_bold").pack(side=tk.LEFT)
        self.machine_combo = StyledCombobox(
            machine_frame,
            values=["DESF", "PICADORA", "TODAS"]
        )
        self.machine_combo.pack(side=tk.LEFT, padx=(PADDING["medium"], 0))
        
        # Method selection
        method_frame = StyledLabelFrame(content_frame, text="Método de Detección")
        method_frame.pack(fill=tk.X, pady=PADDING["medium"])
        
        method_inner = StyledFrame(method_frame)
        method_inner.pack(fill=tk.X, pady=PADDING["small"])
        
        StyledLabel(method_inner, text="Método:", style="body_bold").pack(side=tk.LEFT)
        self.method_combo = StyledCombobox(
            method_inner,
            values=[
                "combined (ensamble por votación, recomendado)",
                "zscore (z-score estándar)",
                "modified_zscore (basado en mediana)",
                "iqr (rango intercuartílico)",
                "rolling (ventana móvil)",
                "isolation_forest (basado en ML)"
            ],
            width=42
        )
        self.method_combo.pack(side=tk.LEFT, padx=(PADDING["medium"], 0))
        
        # Button frame
        btn_frame = StyledFrame(content_frame)
        btn_frame.pack(fill=tk.X, pady=PADDING["large"])
        
        # Run button
        self.run_btn = StyledButton(
            btn_frame,
            text="▶ Detectar Anomalías",
            command=self._run_detection,
            style="primary",
            width=25
        )
        self.run_btn.pack(anchor=tk.CENTER)
        
        # Output info
        StyledLabel(
            content_frame,
            text="Salida: ./data/anomaly y visualizaciones locales en anomaly_results/{MACHINE}/",
            style="small"
        ).pack(anchor=tk.CENTER, pady=PADDING["medium"])
        
        # Copyright
        ttk.Separator(content_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=PADDING["large"]
        )
        StyledLabel(
            content_frame,
            text="© IDC Ingeniería de Confiabilidad | Riopaila Castilla",
            style="small",
            fg=COLORS["text_muted"]
        ).pack(anchor=tk.CENTER, pady=PADDING["small"])
    
    def _run_detection(self):
        if self._running:
            return
        
        self.set_running(True)
        self.run_btn.set_enabled(False)
        self.set_status("Detectando anomalías...", "info")
        
        def run():
            try:
                self.log("=" * 60, "heading")
                self.log("Iniciando Detección de Anomalías", "heading")
                self.log("=" * 60, "heading")
                
                import sys
                import os
                from pathlib import Path
                
                project_root = Path(__file__).resolve().parent.parent
                sys.path.insert(0, str(project_root))
                os.chdir(project_root)
                
                from anomaly import AnomalyDetector, AnomalyConfig, AnomalyMethod
                import pandas as pd
                import json
                
                # Get method
                method_text = self.method_combo.get()
                method_name = method_text.split(" ")[0]
                
                method_map = {
                    "combined": AnomalyMethod.COMBINED,
                    "zscore": AnomalyMethod.ZSCORE,
                    "modified_zscore": AnomalyMethod.MODIFIED_ZSCORE,
                    "iqr": AnomalyMethod.IQR,
                    "rolling": AnomalyMethod.ROLLING,
                    "isolation_forest": AnomalyMethod.ISOLATION_FOREST
                }
                
                config = AnomalyConfig(method=method_map.get(method_name, AnomalyMethod.COMBINED))
                detector = AnomalyDetector(config)
                
                selected_machine = self.machine_combo.get()
                if selected_machine == "TODAS":
                    machines_to_process = ["DESF", "PICADORA"]
                else:
                    machines_to_process = [selected_machine]
                
                self.log(f"Configuración:", "info")
                self.log(f"  Máquinas: {', '.join(machines_to_process)}", "info")
                self.log(f"  Método: {method_name}", "info")
                self.log("", "info")
                
                # Define pathology variables for filtering
                pathology_vars = {
                    "DESF": [
                        "desbalance_chum_la", "desalineacion_rad_chum_la", "soltura_chum_la",
                        "rodamiento_chum_la", "desalineacion_ang_chum_la",
                        "desbalance_chum_lb", "desalineacion_chum_lb", "soltura_chum_lb",
                        "rodamiento_chum_lb"
                    ],
                    "PICADORA": [
                        "desbalance_chum_la", "desalineacion_rad_chum_la", "soltura_chum_la",
                        "rodamiento_chum_la", "desalineacion_ang_chum_la",
                        "desbalance_chum_ll", "soltura_chum_ll", "rodamiento_chum_ll"
                    ]
                }
                
                for machine in machines_to_process:
                    self.log(f"--- Procesando Máquina: {machine} ---", "heading")
                    
                    processed_dir = project_root / "processed" / machine
                    
                    # Output directories: visuals in anomaly_results/, CSV data in data/anomaly/
                    visuals_output_dir = project_root / "anomaly_results" / machine
                    visuals_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    csv_output_dir = project_root / "data" / "anomaly"
                    csv_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    if not processed_dir.exists():
                        self.log(f"Advertencia: Directorio no encontrado: {processed_dir}", "warning")
                        self.log("Ejecute primero el Pipeline de Rendimiento para generar datos procesados.", "warning")
                        continue
                    
                    parquet_files = list(processed_dir.glob("*.parquet"))
                    self.log(f"Encontrados {len(parquet_files)} archivos parquet", "info")
                    
                    all_anomaly_records = []  # For CSV export
                    
                    for pq_file in parquet_files:
                        var_name = pq_file.stem
                        
                        # Filter for pathology variables only
                        if machine in pathology_vars and var_name not in pathology_vars[machine]:
                            continue
                            
                        self.log(f"Procesando: {var_name}", "info")
                        
                        try:
                            df = pd.read_parquet(pq_file)
                            df_detected, result = detector.detect(df, column="y", return_details=True)
                            
                            # Generate visualization (Always generate for local inspection)
                            plot_path = visuals_output_dir / f"{var_name}_anomalies.png"
                            # Remove existing file if it exists to ensure replacement
                            if plot_path.exists():
                                try:
                                    plot_path.unlink()
                                except:
                                    pass
                                    
                            detector.plot_anomalies(
                                df_detected,
                                column="y",
                                save_path=str(plot_path),
                                title=f"Anomaly Detection: {var_name} ({machine})"
                            )
                            import matplotlib.pyplot as plt
                            plt.close()
                            
                            if result.n_anomalies > 0:
                                self.log(f"  Encontradas {result.n_anomalies} anomalías ({result.anomaly_pct:.2f}%)", "info")
                                
                                # Collect anomaly data for CSV export (Power BI)
                                anomaly_df = df_detected[df_detected["is_anomaly"]].copy()
                                
                                # Remove internal columns if they exist
                                cols_to_drop = ["y_clean", "was_interpolated"]
                                anomaly_df = anomaly_df.drop(columns=[c for c in cols_to_drop if c in anomaly_df.columns])
                                
                                anomaly_df["variable"] = var_name
                                anomaly_df["machine"] = machine
                                anomaly_df["method"] = method_name
                                all_anomaly_records.append(anomaly_df)
                            else:
                                self.log(f"  Sin anomalías detectadas", "info")
                            
                        except Exception as e:
                            self.log(f"  Error: {str(e)}", "error")
                    
                    # Export CSV for Power BI consumption
                    if all_anomaly_records:
                        try:
                            detailed_df = pd.concat(all_anomaly_records, ignore_index=True)
                            
                            # Establish specific column order
                            desired_cols = ["timestamp", "y", "is_anomaly", "variable", "machine", "method"]
                            final_cols = [c for c in desired_cols if c in detailed_df.columns]
                            detailed_df = detailed_df[final_cols]
                            
                            detailed_csv_path = csv_output_dir / f"ANOMALIES_{machine}.csv"
                            detailed_df.to_csv(detailed_csv_path, index=False)
                            self.log(f"CSV de anomalías guardado: {detailed_csv_path}", "success")
                            self.log(f"  Total registros de anomalías: {len(detailed_df)}", "info")
                        except PermissionError as pe:
                            self.log(f"❌ Error de Permiso: No se pudo guardar el archivo CSV.", "error")
                            self.log(f"   Archivo bloqueado: {pe.filename}", "error")
                            self.log("⚠️  SOLUCIÓN: Cierre el archivo en Excel/Power BI y vuelva a intentar.", "warning")
                        except Exception as e:
                            self.log(f"❌ Error al guardar CSV: {str(e)}", "error")
                    else:
                        self.log(f"No se encontraron anomalías para exportar en {machine}.", "info")
                
                self.log("=" * 60, "heading")
                self.log("✅ ¡Detección de anomalías completada!", "success")
                self.log("=" * 60, "heading")
                self.set_status("Detección de anomalías completada", "success")
                
            except Exception as e:
                self.log(f"❌ Error: {str(e)}", "error")
                import traceback
                self.log(traceback.format_exc(), "error")
                self.set_status(f"Error: {str(e)}", "error")
            finally:
                self.set_running(False)
                self.run_btn.set_enabled(True)
        
        threading.Thread(target=run, daemon=True).start()


class AboutPanel(StyledFrame):
    """About panel with information about the application."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._images = {}  # Store image references to prevent garbage collection
        self._setup_ui()
    
    def _setup_ui(self):
        # Create scrollable container for all content
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill=tk.BOTH, expand=True)
        
        # Use scrollable.interior as the content frame
        content_frame = scrollable.interior
        
        # Logo images container
        logo_frame = StyledFrame(content_frame)
        logo_frame.pack(pady=(PADDING["xlarge"], PADDING["medium"]))
        
        # Load logos side by side
        try:
            from PIL import Image, ImageTk
            from pathlib import Path
            
            # Riopaila logo
            riopaila_logo_path = Path(__file__).resolve().parent / "assets" / "riopaila_logo.png"
            if riopaila_logo_path.exists():
                img = Image.open(riopaila_logo_path)
                aspect_ratio = img.width / img.height
                new_height = 80
                new_width = int(new_height * aspect_ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                self._images["riopaila_logo"] = ImageTk.PhotoImage(img)
                
                riopaila_label = tk.Label(
                    logo_frame,
                    image=self._images["riopaila_logo"],
                    bg=COLORS["bg_dark"]
                )
                riopaila_label.pack(side=tk.LEFT, padx=PADDING["medium"])
            
            # Logo.ico
            idc_logo_path = Path(__file__).resolve().parent / "assets" / "Logo.ico"
            if idc_logo_path.exists():
                img = Image.open(idc_logo_path)
                aspect_ratio = img.width / img.height
                new_height = 80
                new_width = int(new_height * aspect_ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                self._images["idc_logo"] = ImageTk.PhotoImage(img)
                
                idc_label = tk.Label(
                    logo_frame,
                    image=self._images["idc_logo"],
                    bg=COLORS["bg_dark"]
                )
                idc_label.pack(side=tk.LEFT, padx=PADDING["medium"])
        except Exception as e:
            # If image loading fails, continue without logos
            pass
        
        # Logo/Title
        StyledLabel(
            content_frame,
            text="MainDataForecaster",
            style="title"
        ).pack(pady=(0, PADDING["medium"]))
        
        StyledLabel(
            content_frame,
            text="Predicción de Rendimiento de equipos críticos (Picadora y Desfibradora)\ny de anomalías por medio de modelos N-BEATS y técnicas estadísticas.",
            style="subheading"
        ).pack()
        
        StyledLabel(
            content_frame,
            text="Versión 1.0.0",
            style="small"
        ).pack(pady=PADDING["small"])
        
        # Separator
        ttk.Separator(content_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=PADDING["large"], padx=PADDING["xlarge"]
        )
        
        # Features
        features_text = """
Funcionalidades:
  • Pipeline de Rendimiento - Calcular índices EPI y PPI
  • Entrenamiento N-BEATS - Entrenar modelos de pronóstico con deep learning
  • Pronósticos N-BEATS - Generar pronósticos multi-horizonte
  • Detección de Anomalías - Detectar y limpiar anomalías en datos de sensores

Equipos Soportados:
  • Picadora (20+ variables de sensores)
  • Desfibradora (22+ variables de sensores)
  • Nivel de planta (rendimiento combinado)

Horizontes de Pronóstico:
  • 2 días  - Alta confiabilidad (~85% precisión)
  • 5 días  - Confiabilidad media (~72% precisión)
  • 15 días - Baja confiabilidad (~55% precisión)
  • 1 mes   - Indicación de tendencia (~42% precisión)
        """
        
        StyledLabel(
            content_frame,
            text=features_text,
            style="body",
            justify=tk.LEFT
        ).pack(anchor=tk.W, padx=PADDING["xlarge"])
        
        # Separator
        ttk.Separator(content_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=PADDING["large"], padx=PADDING["xlarge"]
        )
        
        # Credits
        StyledLabel(
            content_frame,
            text="© IDC Ingeniería de Confiabilidad | Riopaila Castilla",
            style="small"
        ).pack(pady=PADDING["medium"])


class SharePointPanel(BasePanel):
    """Panel for SharePoint Synchronization operations."""
    
    def __init__(self, parent, log_callback: Callable, status_callback: Callable, **kwargs):
        super().__init__(parent, log_callback, status_callback, **kwargs)
        self._setup_ui()
    
    def _setup_ui(self):
        # Create scrollable container for all content
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill=tk.BOTH, expand=True)
        
        # Use scrollable.interior as the content frame
        content_frame = scrollable.interior
        
        # Title
        StyledLabel(
            content_frame, 
            text="🔄 Sincronización SharePoint",
            style="heading"
        ).pack(anchor=tk.W, pady=(0, PADDING["medium"]))
        
        # Description
        StyledLabel(
            content_frame,
            text="Automatizar la sincronización de datos con SharePoint.\n"
                 "Descargar datos crudos y subir pronósticos generados.",
            style="body"
        ).pack(anchor=tk.W, pady=(0, PADDING["large"]))
        
        # Options frame
        options_frame = StyledLabelFrame(content_frame, text="Opciones de Sincronización")
        options_frame.pack(fill=tk.X, pady=PADDING["medium"])
        
        # Checkboxes
        self.download_var = tk.BooleanVar(value=True)
        StyledCheckbutton(
            options_frame,
            text="Descargar datos crudos desde SharePoint",
            variable=self.download_var
        ).pack(anchor=tk.W, pady=PADDING["small"])
        
        self.upload_var = tk.BooleanVar(value=True)
        StyledCheckbutton(
            options_frame,
            text="Subir pronósticos a SharePoint",
            variable=self.upload_var
        ).pack(anchor=tk.W, pady=PADDING["small"])
        
        self.upload_anomaly_var = tk.BooleanVar(value=True)
        StyledCheckbutton(
            options_frame,
            text="Subir resultados de anomalías a SharePoint",
            variable=self.upload_anomaly_var
        ).pack(anchor=tk.W, pady=PADDING["small"])
        
        self.replace_latest_var = tk.BooleanVar(value=True)
        StyledCheckbutton(
            options_frame,
            text="Reemplazar archivos existentes (Forecast y Anomalías)",
            variable=self.replace_latest_var
        ).pack(anchor=tk.W, pady=(0, PADDING["small"]), padx=PADDING["large"])
        
        # Clarifying message
        StyledLabel(
            options_frame,
            text="ℹ Active 'Descargar' si necesita inicializar datos locales.\n"
                 "   Active 'Subir' si ya tiene resultados locales para sincronizar.",
            style="small",
            fg=COLORS["text_secondary"]
        ).pack(anchor=tk.W, pady=(PADDING["small"], PADDING["medium"]), padx=PADDING["medium"])
        
        # Button frame
        btn_frame = StyledFrame(content_frame)
        btn_frame.pack(fill=tk.X, pady=PADDING["large"])
        
        # Run button
        self.run_btn = StyledButton(
            btn_frame,
            text="▶ Ejecutar Sincronización",
            command=self._run_sync,
            style="primary",
            width=30
        )
        self.run_btn.pack(anchor=tk.CENTER)
        
        # Output info
        StyledLabel(
            content_frame,
            text="Rutas de SharePoint:\n"
                 "  • Descarga: Mantenimiento Predictivo Castilla/Fuentes de Datos/{Performance, Dimensiones}\n"
                 "  • Subida Forecast: .../Fuentes de Datos/Forecast/{Fore_Desfibradora...}\n"
                 "  • Subida Anomalías: .../Fuentes de Datos/Anomalías",
            style="small"
        ).pack(anchor=tk.CENTER, pady=PADDING["medium"])
        
        # Copyright
        ttk.Separator(content_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=PADDING["large"]
        )
        StyledLabel(
            content_frame,
            text="© IDC Ingeniería de Confiabilidad | Riopaila Castilla",
            style="small",
            fg=COLORS["text_muted"]
        ).pack(anchor=tk.CENTER, pady=PADDING["small"])

    def _run_sync(self):
        if self._running:
            return
        
        if not self.download_var.get() and not self.upload_var.get() and not self.upload_anomaly_var.get():
            messagebox.showwarning("Advertencia", "Seleccione al menos una opción de sincronización.")
            return

        self.set_running(True)
        self.run_btn.set_enabled(False)
        self.set_status("Iniciando sincronización con SharePoint...", "info")
        
        def run():
            try:
                self.log("=" * 60, "heading")
                self.log("Iniciando Sincronización SharePoint", "heading")
                self.log("=" * 60, "heading")
                
                import sys
                import os
                from pathlib import Path
                
                # Add project root to path
                project_root = Path(__file__).resolve().parent.parent
                sys.path.insert(0, str(project_root))
                os.chdir(project_root)
                
                from common.sharepoint_downloader import SharePointDownloader
                from common.sharepoint_uploader import SharePointUploader
                
                # Check credentials first
                import os
                if not all([os.getenv("SHAREPOINT_SITE_URL"), os.getenv("SHAREPOINT_CLIENT_ID"), os.getenv("SHAREPOINT_CLIENT_SECRET"), os.getenv("SHAREPOINT_TENANT_ID")]):
                    self.log("⚠ Credenciales de SharePoint no encontradas en variables de entorno.", "warning")
                    self.log("   Configure SHAREPOINT_SITE_URL, SHAREPOINT_CLIENT_ID, SHAREPOINT_CLIENT_SECRET y SHAREPOINT_TENANT_ID en .env.", "warning")
                
                # 1. Download Phase
                if self.download_var.get():
                    self.log("\n📥 Fase 1: Descarga de Datos", "subheading")
                    downloader = SharePointDownloader()
                    if downloader.connect():
                        self.log("Conectado a SharePoint para descarga.", "success")
                        results = downloader.sync_data(project_root)
                        total_files = sum(results.values())
                        self.log(f"Descarga completada. Total archivos: {total_files}", "info")
                    else:
                        self.log("❌ Falló la conexión para descarga.", "error")
                        if getattr(downloader, "last_error_details", None):
                            self.log(downloader.last_error_details, "error")



                # 3. Upload Phase (Forecasts)
                if self.upload_var.get():
                    self.log("\n📤 Fase 3: Subida de Pronósticos", "subheading")
                    uploader = SharePointUploader()
                    if uploader.connect():
                        self.log("Conectado a SharePoint para subida.", "success")
                        
                        # Define upload mappings: Local Folder -> Remote Folder
                        # Forecasts are generated in data/forecasts/{MACHINE}/
                        # Visuals are generated in forecast_visuals/{MACHINE}/ (Local only)
                        
                        forecast_base_dir = project_root / "data" / "forecasts"
                        
                        files_to_upload = []
                        
                        # 1. Collect Forecast CSVs ONLY (Summary files, not individual variable forecasts)
                        if forecast_base_dir.exists():
                            # Only upload files starting with FORECAST_
                            files_to_upload.extend(list(forecast_base_dir.rglob("FORECAST_*.csv")))

                            # Special case: PLANT forecast files are sometimes saved without FORECAST_ prefix
                            plant_dir = forecast_base_dir / "PLANT"
                            if plant_dir.exists():
                                plant_forecasts = list(plant_dir.glob("*_nbeats_forecast_*.csv"))
                                if plant_forecasts:
                                    # Avoid duplicates if a FORECAST_ file already exists for PLANT
                                    if not any(p.name.lower().startswith("forecast_plant") for p in files_to_upload):
                                        files_to_upload.extend(plant_forecasts)
                        else:
                            self.log(f"⚠ Directorio de pronósticos no encontrado: {forecast_base_dir}", "warning")
                            
                        if not files_to_upload:
                            self.log(f"⚠ No se encontraron archivos para subir.", "warning")
                        
                        # Group files by target folder
                        files_by_folder = {}
                        
                        for file_path in files_to_upload:
                            filename = file_path.name.lower()
                            remote_subfolder = None
                            
                            # Determine target folder based on filename or parent folder
                            parent_name = file_path.parent.name.lower()
                            
                            if "desf" in filename or "desf" in parent_name: 
                                remote_subfolder = "Fore_Desfibradora"
                            elif "picadora" in filename or "picadora" in parent_name: 
                                remote_subfolder = "Fore_Picadora"
                            elif "plant" in filename or "plant" in parent_name:
                                remote_subfolder = "Fore_Planta"
                            
                            if remote_subfolder:
                                # Update path to include Mantenimiento Predictivo Castilla
                                remote_path = f"Mantenimiento Predictivo Castilla/Fuentes de Datos/Forecast/{remote_subfolder}"
                                if remote_path not in files_by_folder:
                                    files_by_folder[remote_path] = []
                                files_by_folder[remote_path].append(file_path)
                            else:
                                # Skip files that don't match a machine pattern
                                pass
                        
                        # Process each folder
                        for remote_path, folder_files in files_by_folder.items():
                            # Clear folder if requested
                            if self.replace_latest_var.get():
                                self.log(f"Limpiando carpeta remota: {remote_path}...", "info")
                                # Use try/except to avoid crashing if folder doesn't exist
                                try:
                                    if uploader.delete_folder_contents(remote_path):
                                        self.log(f"✅ Carpeta limpiada: {remote_path}", "success")
                                    else:
                                        self.log(f"⚠ Nota: No se pudo limpiar carpeta (quizás está vacía)", "warning")
                                except Exception:
                                    pass # Ignore 404s during delete
                            
                            # Upload files
                            for file_path in folder_files:
                                self.log(f"Subiendo {file_path.name} a {remote_path}...", "info")
                                if uploader.upload_file(file_path, remote_path):
                                    self.log(f"✅ Subido: {file_path.name}", "success")
                                else:
                                    self.log(f"❌ Error subiendo: {file_path.name}", "error")
                                    if getattr(uploader, "last_error_details", None):
                                        self.log(uploader.last_error_details, "error")
                    else:
                        self.log("❌ Falló la conexión para subida.", "error")
                        if getattr(uploader, "last_error_details", None):
                            self.log(uploader.last_error_details, "error")

                # 4. Upload Phase (Anomalies)
                if self.upload_anomaly_var.get():
                    self.log("\n📤 Fase 4: Subida de Anomalías", "subheading")
                    uploader = SharePointUploader()
                    if uploader.connect():
                        self.log("Conectado a SharePoint para subida de anomalías.", "success")
                        
                        anomaly_base_dir = project_root / "data" / "anomaly"
                        if not anomaly_base_dir.exists():
                            self.log(f"⚠ Directorio de anomalías no encontrado: {anomaly_base_dir}", "warning")
                        else:
                            # Find all anomaly CSVs
                            anomaly_files = list(anomaly_base_dir.glob("ANOMALIES_*.csv"))
                            if not anomaly_files:
                                self.log("⚠ No se encontraron archivos de anomalías para subir.", "warning")
                            else:
                                # Fixed path with parent folder
                                remote_anomaly_path = "Mantenimiento Predictivo Castilla/Fuentes de Datos/Anomalías"
                                
                                # Clear folder if requested
                                if self.replace_latest_var.get():
                                    self.log(f"Limpiando carpeta remota: {remote_anomaly_path}...", "info")
                                if self.replace_latest_var.get():
                                    self.log(f"Limpiando carpeta remota: {remote_anomaly_path}...", "info")
                                    try:
                                        if uploader.delete_folder_contents(remote_anomaly_path):
                                            self.log(f"✅ Carpeta limpiada: {remote_anomaly_path}", "success")
                                    except Exception:
                                        pass
                                
                                for file_path in anomaly_files:
                                    self.log(f"Subiendo {file_path.name} a {remote_anomaly_path}...", "info")
                                    if uploader.upload_file(file_path, remote_anomaly_path):
                                        self.log(f"✅ Subido: {file_path.name}", "success")
                                    else:
                                        self.log(f"❌ Error subiendo: {file_path.name}", "error")

                self.log("=" * 60, "heading")
                self.log("✅ Sincronización finalizada.", "success")
                self.set_status("Sincronización finalizada", "success")
                
            except Exception as e:
                self.log(f"❌ Error crítico: {str(e)}", "error")
                self.set_status(f"Error: {str(e)}", "error")
            finally:
                self.set_running(False)
                self.run_btn.set_enabled(True)
        
        threading.Thread(target=run, daemon=True).start()
