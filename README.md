# IDC RIOP - Industrial Machinery Forecasting

> N-BEATS deep learning for predictive maintenance of Desfibradora and Picadora equipment.

---

## 🚀 Quick Start - User Interface

The recommended way to use the pipeline is through the graphical user interface.

### 1. Installation & Launch

```powershell
# Create and activate virtual environment (if not already done)
python -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch the UI
python launch_app.py
```

### 2. Workflow via UI

The application is organized into tabs for each step of the process:

#### 🔄 1. SharePoint Sync (Data Ingestion)
- Go to the **SharePoint** tab.
- Click **"Descargar datos crudos"** to fetch the latest performance (EPI) and dimension data.
- **Note**: Raw sensor data (`DATOS_DESF_CLEANED.csv`, `DATOS_PICADORA_CLEANED.csv`) should be placed manually in the `data/` folder if not available on SharePoint.

#### 🧠 2. Training
- Go to the **Training** tab.
- Select a **Machine** (DESF, PICADORA, or ALL).
- Choose **Aggregation**:
  - `Mean`: Best for trends and general forecasting.
  - `Max`: Best for peak detection and alarm thresholds.
- Click **"Iniciar Entrenamiento"**.
- *Tip*: Use "Forzar reentrenamiento" to ensure models use the latest data.

#### 📈 3. Forecasting & Visualization
- Go to the **Forecast** tab.
- Select **Machine** and **Horizon** (e.g., `2_days`, `5_days`).
- Click **"Generar Pronósticos"**.
- **Local Visualization**: 
  - Click the **"👁 Ver"** button next to the horizon selector to see an interactive chart of data coverage vs. forecast horizon.
  - Detailed plots are saved to `forecast_visuals/{MACHINE}/`.

#### 🔍 4. Anomaly Detection
- Go to the **Anomaly** tab.
- Detect and clean anomalies in sensor data before training/forecasting.
- Results are saved to `anomaly_results/` and `data/anomaly/`.

---

## 📂 Data & Output Locations

### Input Data
- **Raw Sensor Data**: Place your CSV files in `data/` (e.g., `data/DATOS_DESF_CLEANED.csv`).
- **EPI/Performance Data**: Synced from SharePoint to `data/epi/`.

### Outputs
| Content | Location | Description |
|---------|----------|-------------|
| **Forecast CSVs** | `data/forecasts/{MACHINE}/` | Combined forecast files (e.g., `FORECAST_DESF_2_DAYS.csv`). |
| **Visualizations** | `forecast_visuals/{MACHINE}/` | Individual plots (`.png`) and interactive charts. |
| **Anomaly Reports** | `anomaly_results/{MACHINE}/` | Anomaly detection plots and summaries. |
| **Power BI Data** | `data/anomaly/` | CSV exports formatted for Power BI dashboards. |

---

## 💻 Command Line Interface (Advanced)

For automation or server environments, you can still use the CLI.

### Complete Pipeline
```powershell
# Train and forecast all machines
python run_nbeats_pipeline.py

# Train and forecast specific machine
python run_nbeats_pipeline.py --machine PICADORA
```

### Training Only
```powershell
# Train all models (MEAN aggregation - recommended)
python -m nbeats.nbeats_train --batch --machine ALL --force --agg mean

# Train with MAX aggregation (for peak detection)
python -m nbeats.nbeats_train --batch --machine ALL --force --agg max
```

### Forecasting Only
```powershell
# 2-day forecast (48h)
python -m nbeats.nbeats_forecast --batch --horizon 2_days

# 5-day forecast (120h)
python -m nbeats.nbeats_forecast --batch --horizon 5_days
```

### Anomaly Detection
```powershell
# Run detection on a specific file
python -m anomaly.anomaly_detector --input processed/DESF/corriente_motor_a.parquet
```

---

## 🔧 Project Structure

```
IDC_RIOP/
├── launch_app.py           # UI Entry point
├── ui/                     # User Interface code
├── data/                   # Input/output data
│   ├── epi/                # Performance data (from SharePoint)
│   ├── forecasts/          # Generated forecast CSVs
│   └── raw/                # Raw sensor data
├── forecast_visuals/       # Forecast plots (PNG/HTML)
├── anomaly_results/        # Anomaly detection reports
├── models/                 # Trained N-BEATS models
├── nbeats/                 # N-BEATS core logic
├── common/                 # Shared utilities (SharePoint, Data Prep)
└── requirements.txt        # Python dependencies
```

---

## 📋 Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`
- SharePoint credentials (in `.env`) for data sync features.

