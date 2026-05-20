# AURORA 2.0 – Adaptive Mining Activity Monitoring

Sentinel-2 + Sentinel-1 SAR Fusion | Google Earth Engine | Streamlit Dashboard

---

## Overview

AURORA 2.0 is a mine-adaptive excavation monitoring system using:

- Sentinel-2 optical imagery for excavation detection
- Sentinel-1 SAR imagery for cloud-gap filling

The system is fully unsupervised and learns excavation signatures independently for each mine.

---

# Repository Structure

- `pipelines.py`  
  Core training, monitoring, SAR fusion, and inference logic.

- `outputs.py`  
  Plot generation and report exports.

- `ui/app.py`  
  Streamlit dashboard.

- `testCode.ipynb`  
  Notebook for drawing No-Go Zones.

- `shpToGJson.ipynb`  
  Shapefile → GeoJSON conversion utility.

---

# Reports & Documentation

## `MidTermReport.pdf`
Midterm report submitted during the AURORA 2.0 event.

## `EndTermReport.pdf`
Final endterm report containing the complete pipeline and results.

## `Aurora_2.0_Marauders.pptx`
Presentation slides used during the final project presentation/demo.

## `Aurora_2.0_Updates.pdf`
Post-event update document containing SAR integration and other improvements

---

# Installation

Install required packages:

```bash
pip install earthengine-api geemap geopandas rasterio scikit-learn numpy pandas matplotlib joblib leafmap streamlit
```

Google Earth Engine authentication is required.

---

# Example Folder Structure

```text
Mine Data/
└── Mine_226_Data/
    ├── Outputs/
    ├── clusterData.json
    ├── kmeans.pkl
    ├── scaler.pkl
    ├── mine_226_features_0.csv
    ├── mine_226_excavation_1.tif
    ├── mine_226_excavation_7.tif
    ├── mine_226_excavation_9.tif
    ├── mine_226_sar_1.tif
    ├── mine_226_sar_7.tif
    ├── mine_226_sar_9.tif
    └── nogozones.geojson
```

---

# Dashboard

Run the dashboard using:

```bash
python -m streamlit run ui/app.py
```

---

# Dashboard Features

- Mine selector
- Spatial excavation maps
- Confidence heatmaps
- SAR analysis plots
- Area vs Time plots
- Alert logs
- CSV report viewer

---

# ⭐ Starred Mines

Some mines appear with a ⭐ in the dropdown.

These are mines that are fully updated and contain:
- latest SAR integration
- latest outputs
- latest analysis features
---

# Workflow

You can use `testCode.ipynb`

## 1) Training

### Export training features

```python
pipelines.trainingStart(
    "2020-01-01",
    "2022-01-01",
    226,
    60,
    debug=0
)
```

Download the exported CSV from Google Drive and place it inside:

```text
Mine Data/Mine_226_Data/
```

### Train the model

```python
pipelines.trainingComplete(226, debug=0)
```

This generates:
- `kmeans.pkl`
- `scaler.pkl`
- `clusterData.json`

---

## 2) Optical Monitoring (Sentinel-2)

Large timelines should be split using debug values.

```python
pipelines.monitoringStart(
    "2017-01-01",
    "2022-01-01",
    226,
    60,
    debug=1
)

pipelines.monitoringStart(
    "2022-01-01",
    "2024-01-01",
    226,
    60,
    debug=7
)

pipelines.monitoringStart(
    "2024-01-01",
    "2026-01-01",
    226,
    60,
    debug=9
)
```

Download all exported GeoTIFFs and place them inside:

```text
Mine Data/Mine_226_Data/
```

---

## 3) SAR Monitoring (Sentinel-1)

Use the SAME debug values as optical monitoring.

```python
pipelines.sarMonitoringStart(
    "2017-01-01",
    "2022-01-01",
    226,
    debug=1
)

pipelines.sarMonitoringStart(
    "2022-01-01",
    "2024-01-01",
    226,
    debug=7
)

pipelines.sarMonitoringStart(
    "2024-01-01",
    "2026-01-01",
    226,
    debug=9
)
```

Download all exported SAR GeoTIFFs and place them inside:

```text
Mine Data/Mine_226_Data/
```

---

## 4) Final Monitoring + SAR Fusion

```python
pipelines.monitoringComplete(
    226,
    60,
    debug=[1, 7, 9],
    sar_debug=[1, 7, 9]
)
```

This:
- stitches monitoring chunks
- stitches SAR chunks
- performs SAR-assisted gap filling
- generates plots
- generates reports
- generates alerts

Outputs are saved inside:

```text
Mine Data/Mine_226_Data/Outputs/
```

---

# No-Go Zones

Use `testCode.ipynb` to draw restricted regions interactively.

The notebook exports:

```text
nogozones.geojson
```

Once present, the monitoring pipeline automatically generates:
- violation alerts
- restricted-area overlays

---

# Current Limitations

- Sentinel-2 has 10m spatial resolution
- SAR is sensitive to moisture and seasonal variation
- Long monitoring periods require manual chunking
- Persistent cloud cover still affects optical monitoring quality

---

# Future Scope

- Better seasonal SAR modeling
- Multi-sensor fusion
- Semi-supervised refinement
- Real-time monitoring automation
- Improved temporal change detection

---

# License

This project is intended for academic and research purposes.