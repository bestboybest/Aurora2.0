# AURORA 2.0 – Adaptive Mining Activity Monitoring

## Sentinel-2 Time-Series | Google Earth Engine | Unsupervised Learning
---
## Overview
This repository implements AURORA 2.0 adaptive mining monitoring pipeline, an unsupervised, mine-specific system for the purpose of detecting and monitoring excavation activity using Sentinel-2 time-series imagery.
The system learns the spectral signature of excavation for each mine individually which enables strong performance across various types of mines and differing seasonal conditions.

### Pipeline
- Learns excavation signatures from historical data
- Detects newly excavated pixels over time
- Produces time-resolved binary excavation masks
- Is scalable to large mine inventories 
---
## Inputs
- Sentinel-2 Level-2A Multispectral Time Series Data obtained from Google Earth Engine
via Python API to streamline the workflow.
- Geometries of Legal Mine Boundaries for all mines under consideration.
- No-Go Zone Polygons synthetically generated to demonstrate model effectiveness
---
### Spectral Indices Used
- NDVI – Normalized Difference Vegetation Index (effective for detecting vegetation removal associated with mining)
- SWIR – Shortwave Infrared (capable of detecting exposed rocks and soil resulting from
mining operations)
- NBR – Normalized Burn Ratio (changes in NBR can indicate surface disturbances)
- BSI – Bare Soil Index (mining activities expose bare soil, which this index can highlight)
---
## Method Summary
- Filtration of Sentinel-2 images by mine boundaries and cloud-masked using SCL
- Extraction of spectral and temporal features
- Features are standardized and clustered using K-Means Clustering
- Identification of excavation clusters via centroid meta-clustering 
- Classification of pixels as excavated or non-excavated over time
- Outputs obtained using morphological post-processing
---
