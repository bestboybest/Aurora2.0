# AURORA 2.0 – Adaptive Mining Activity Monitoring

**Sentinel-2 Time Series | Google Earth Engine | Unsupervised Learning | Streamlit Dashboard**

---

## Overview
AURORA 2.0 is an adaptive, mine-specific monitoring system designed to detect and track excavation activity using Sentinel-2 multispectral time-series imagery.

The pipeline is fully unsupervised and learns excavation signatures independently for each mine. This allows it to generalize across diverse mine types, geographies, and seasonal conditions without requiring labeled training data.

---

## Key Capabilities
- **Mine-specific excavation signature learning:** Automatically adapts to local mineralogy and soil types.
- **Time-resolved excavation detection:** Tracks progress over daily/monthly intervals using a "Confidence System" to filter noise.
- **No-Go Zone monitoring:** Interactively define restricted zones (e.g., forests, water bodies) and receive hierarchical alerts for intrusions.
- **Resilient Data Handling:** Capable of processing large, chunked GeoTIFF exports from Earth Engine.
- **Interactive Dashboard:** Built-in Streamlit UI for visualizing results without writing code.

---

## Repository Contents
- `pipelines.py` – Core logic for training (clustering) and monitoring (inference).
- `outputs.py` – Visualization generation, plotting, and alert system logic.
- `dashboard.py` – Interactive Streamlit UI code.
- `testCode.ipynb` – Notebook for interactively drawing restricted zones.
- `Mine_Data/` – (User Created) Directory storing per-mine data, trained models (`.pkl`), and outputs.

---

## Inputs & Data Requirements
1.  **Sentinel-2 Level-2A imagery:** Accessed automatically via the Google Earth Engine Python API.
2.  **Mine boundary polygons:** A shapefile (`.shp`) containing legal mine extents.
3.  **No-Go Zone polygons (Optional):** User-defined GeoJSON files created via the provided notebook.

---

## Installation & Setup

### 1. Python Environment
Google Earth Engine authentication is required. Ensure the following packages are installed:

pip install earthengine-api geemap geopandas rasterio scikit-learn numpy pandas matplotlib joblib leafmap streamlit

### 2. Directory Structure
You must manually create a directory structure for your specific mine before running the code.

Example for Mine ID `226`:
# Create the main folder and the outputs subfolder
mkdir -p "Mine Data/Mine_226_Data/Outputs"

The system expects files to be placed here after downloading them from Google Drive (see Usage below).

---

## Usage Workflow

The pipeline is split into **Training**, **Monitoring**, and **Analysis (UI)**.

### Phase 1: Training a Mine
1.  **Run the Export Script:**
    Call `trainingStart(startTime, endTime, mineid, windowSize)` in your Python environment.
    * *Action:* This triggers an Export Task in Google Earth Engine.
2.  **Download Data:**
    Go to your Google Drive. You will find a CSV file named `mine_226_features_0.csv`.
3.  **Place File:**
    Move this CSV into your local folder: `Mine Data/Mine_226_Data/`.
4.  **Train the Model:**
    Run `trainingComplete(mineid)`.
    * *Result:* This saves `kmeans.pkl`, `scaler.pkl`, and `clusterData.json` into the folder.

### Phase 2: Monitoring a Mine
1.  **Run the Inference Script (Manual Splitting):**
    For long monitoring periods, GEE may time out. You should run `monitoringStart` multiple times with different `debug` values (and optionally split your time range) to generate separate chunks.
    * *Example:*
        `monitoringStart('2020-01-01', '2022-01-01', mineid, window, debug=0)`
        `monitoringStart('2022-01-01', '2024-01-01', mineid, window, debug=1)`
    * *Action:* This triggers multiple Export Tasks in GEE.

2.  **Download Data:**
    Go to your Google Drive. You will find GeoTIFF files like `mine_226_excavation_0.tif`, `mine_226_excavation_1.tif`, etc.

3.  **Place Files:**
    Move all TIF files into: `Mine Data/Mine_226_Data/`.

4.  **Generate Outputs:**
    Run `monitoringComplete(mineid, threshold, debug=[0, 1])`.
    * *Stitching:* The `debug` parameter accepts a list. The system will automatically load, stitch, and sort the data from all provided chunks to create a seamless timeline.
    * *Result:* All plots, maps, and logs are generated in the `Outputs/` subfolder.

---

## No-Go Zone Creation (Interactive)

To enable alerts for restricted areas, you must manually define them using the provided Jupyter Notebook (`testCode.ipynb`).

**Steps:**
1.  **Open Notebook:** Launch `testCode.ipynb` in Jupyter, and scroll down to find the line that starts with "mineId = " and insert the mineId of the mine you want to generate no go zone polygon for.
2.  **Load Mine:** The notebook loads the specific mine polygon (e.g., ID 226) and centers the map using `leafmap`.
3.  **Draw Zones:** Use the polygon drawing tool on the map to outline restricted areas (forest buffers, water bodies).
4.  **Save:** Run the export cell.
    * *Result:* This saves `nogozones.geojson` directly to your `Mine Data/Mine_226_Data/` folder.

*Once this file exists, `monitoringComplete` will automatically detect it and generate violation alerts.*

---

## Dashboard (User Interface)

AURORA 2.0 includes a Streamlit-based dashboard to visualize results without touching code.

**How to Run:**
1.  Open your terminal.
2.  Navigate to the project root.
3.  Run the following command:
    python -m streamlit run ui/app.py

**Features:**
- **Mine Selector:** Automatically detects all available mine folders in `Mine Data/`.
- **Gallery:** Browses generated maps and plots (Spatial Maps, Growth Rates, Confidence Heatmaps).
- **Data Viewer:** Allows viewing and downloading of generated CSV reports (Alert Logs, Excavation Intensity).

---

## Limitations (Technical)

1.  **Resolution Constraints (10m):** Sentinel-2 has a 10m spatial resolution. Small-scale illegal digging or single-truck operations (<100m²) may be mixed with background signals and go undetected.
2.  **Spectral Mimicry:** Deep open-pit shadows can spectrally resemble water bodies (low SWIR). While the clustering logic minimizes this, extreme shadow conditions may still cause transient false positives.
3.  **Confidence Latency:** The system prioritizes precision over speed. A pixel must be excavated for a sustained period (e.g., 30 days) to trigger a "Confirmed" status. This introduces a purposeful delay to filter out cloud artifacts.
4.  **Optical Blindness:** The system relies on optical imagery. During extended monsoon periods with 100% cloud cover, the system cannot update the excavation status until the sky clears.

---

## Future Scope

1.  **SAR Fusion (Sentinel-1):** Integrating Synthetic Aperture Radar (SAR) would enable "all-weather" monitoring, allowing the system to detect excavation coherence changes even through heavy cloud cover.
2.  **Semi-Supervised Loop:** The system's "High Confidence" outputs can serve as pseudo-labels to train a lightweight supervised model (e.g., U-Net), progressively improving the decision boundary over time.
3.  **Dynamic Thresholding:** Instead of a fixed temporal threshold (e.g., 30 days), the system could learn an optimal confidence window per mine based on local cloud statistics and data frequency.
4.  **Sub-Pixel Unmixing:** Implementing spectral unmixing algorithms to estimate the percentage of excavation within a single 10m pixel, potentially allowing detection of smaller-scale encroachments.

---

## License
This project is provided for academic and research purposes.
