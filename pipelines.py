import ee
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime
import rasterio
import geemap
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import joblib
import json
import os

from outputs import *

# -------------------------------------------------------------------
# Feature list used consistently across training and monitoring
# Each feature is chosen to separate excavated vs non-excavated pixels
# -------------------------------------------------------------------
features = [
    "NDVI", "B11", "B12",
    "NDVI_median", "B11_median", "B12_median",
    "BSI_median", "NDMI_median",
    "NDVI_var", "NDMI_var",
    "NBR_slope"
]

# -------------------------------------------------------------------
# Load mine boundary polygons and assign mine_id
# CRS is kept in EPSG:4326 for compatibility with Earth Engine
# -------------------------------------------------------------------
gdf = gpd.read_file("./Mine Polygons/SHP/mines_cils.shp").to_crs("EPSG:4326").reset_index(drop=True)
gdf["mine_id"] = gdf.index
gdf = gdf[["mine_id", "area", "perimeter", "geometry"]]

# -------------------------------------------------------------------
# Debug helpers (used during development and sanity checks)
# -------------------------------------------------------------------
def debug(s2, printS):
    print(printS)
    print(s2.size().getInfo())
    print("\n")

def debug2(Efixed, dates):  
    print("Efixed shape:", Efixed.shape)
    print("Efixed dtype:", Efixed.dtype)

    for t in [0, 1, 2, len(Efixed)//2, len(Efixed)-1]:
        vals, counts = np.unique(Efixed[t], return_counts=True)
        print(f"\nTime {t} ({dates[t]}):")
        print(dict(zip(vals.tolist(), counts.tolist())))

    t = 0
    print("t=0 stats:")
    print("sum:", Efixed[t].sum())
    print("mean:", Efixed[t].mean())
    print("nonzero:", np.count_nonzero(Efixed[t]))

# -------------------------------------------------------------------
# Mask Sentinel-2 pixels using Scene Classification Layer (SCL)
# Removes clouds, cloud shadows, cirrus, and snow/ice
# -------------------------------------------------------------------
def mask(image):
    scl = image.select("SCL")
    invalid = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return image.updateMask(invalid.Not())

# -------------------------------------------------------------------
# Compute core spectral indices used for mining detection
# -------------------------------------------------------------------
def addFeatures(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR")
    bsi = image.expression(
        "((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))",
        {
            'SWIR' : image.select("B11"),
            'RED' : image.select("B4"),
            'NIR' : image.select("B8"),
            'BLUE' : image.select("B2")
        }
    ).rename("BSI")
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")

    return image.addBands([ndvi, nbr, bsi, ndmi]).select(
        ["NDVI", "B11", "B12", "NBR", "BSI", "NDMI"]
    )

# -------------------------------------------------------------------
# Add a time band (days since first image) for temporal regression
# -------------------------------------------------------------------
def addTimeBand(image, t0):
    days = image.date().difference(t0, "day")
    time_band = ee.Image.constant(days).toFloat()
    return image.addBands(time_band.rename("time"))

# -------------------------------------------------------------------
# Compute rolling linear regression slope for a band (used for NBR)
# Negative NBR slope is a strong indicator of surface disturbance
# -------------------------------------------------------------------
def addSlope(s2, windowSize, band):
    def addSlopeActual(image):
        date = ee.Date(image.get("system:time_start"))
        window = s2.filterDate(
            date.advance(-windowSize, "day"), date
        ).select(["time", band])
        count = window.size()

        # If insufficient data, slope is masked
        slope = ee.Image(
            ee.Algorithms.If(
                count.gte(2),
                window.reduce(ee.Reducer.linearFit()).select("scale"),
                ee.Image.constant(0).toFloat().mask(
                    ee.Image.constant(0).toFloat()
                )
            )
        ).rename(f"{band}_slope")

        # Clamp to avoid numerical explosions
        slope = slope.clamp(-0.1, 0.1)
        return image.addBands(slope)

    return addSlopeActual

# -------------------------------------------------------------------
# Compute rolling variance and rolling median features
# Rolling medians are used for robustness against outliers
# -------------------------------------------------------------------
def addRollingStats(s2, windowSize):
    def addRollingStatsActual(image):
        date = ee.Date(image.get("system:time_start"))
        window = s2.filterDate(
            date.advance(-windowSize, "day"), date
        )
        count = window.size()

        dummy_var = ee.Image.constant(0).toFloat()
        dummy_median = ee.Image.constant([0, 0, 0, 0, 0]).toFloat()

        ndviVar = ee.Algorithms.If(
            count.gte(2),
            window.select('NDVI').reduce(ee.Reducer.variance()),
            dummy_var
        )
        ndviVar = ee.Image(ndviVar).rename("NDVI_var") 

        ndmiVar = ee.Algorithms.If(
            count.gte(2),
            window.select('NDMI').reduce(ee.Reducer.variance()),
            dummy_var
        )
        ndmiVar = ee.Image(ndmiVar).rename("NDMI_var")

        median = ee.Algorithms.If(
            count.gte(1),
            window.select(['NDVI', 'B11', 'B12', 'BSI', "NDMI"]).median(),
            dummy_median
        )
        median = ee.Image(median).rename(
            ["NDVI_median", "B11_median", "B12_median", "BSI_median", "NDMI_median"]
        )

        return image.addBands([ndviVar, ndmiVar, median])

    return addRollingStatsActual

# -------------------------------------------------------------------
# Complete feature engineering pipeline
# -------------------------------------------------------------------
def featureEngineering(s2, windowSize):
    s2 = s2.map(addFeatures)
    t0 = ee.Date(s2.first().get("system:time_start"))
    s2 = s2.map(lambda img: addTimeBand(img, t0))
    s2 = s2.map(addRollingStats(s2, windowSize))
    s2 = s2.map(addSlope(s2, windowSize, "NBR"))
    return s2.select(features)

# -------------------------------------------------------------------
# Convert images to tabular samples for clustering (training phase)
# -------------------------------------------------------------------
def imageToTable(mine):
    def imageToTableActual(image):
        return image.sample(region=mine, scale=10, geometries=False)
    return imageToTableActual

# We create an "excavated-score" consisting of having
# high SWIR (B12), high BSI, and low NDVI/NDMI values.
# This helps us identify which cluster corresponds to active excavation.
def findExcavated(centroids, labels):
    b12_idx = features.index("B12_median")  
    bsi_idx = features.index("BSI_median")  
    ndvi_idx = features.index("NDVI_median") 
    ndmi_idx = features.index("NDMI_median")

    scores = []
    
    # We compare the two meta-clusters (0 and 1)
    for group_id in [0, 1]:
        # Get indices of clusters belonging to this meta-cluster
        group_indices = [i for i, x in enumerate(labels) if x == group_id]
        
        if len(group_indices) == 0:
            scores.append(-np.inf)
            continue    

        # Compute mean spectral values for this metacluster
        mean_b12 = np.mean([centroids[i][b12_idx] for i in group_indices])
        mean_bsi = np.mean([centroids[i][bsi_idx] for i in group_indices])
        mean_ndvi = np.mean([centroids[i][ndvi_idx] for i in group_indices])
        mean_ndmi = np.mean([centroids[i][ndmi_idx] for i in group_indices])

        # Excavation heuristic score
        score = (2.0 * mean_b12) + (1.0 * mean_bsi) - (0.5 * mean_ndvi) - (1.0 * mean_ndmi)
        scores.append(score)

    # The metacluster with the higher excavation score is considered excavated
    exLabel = 0 if scores[0] > scores[1] else 1
    return exLabel, scores

# Creates an Earth Engine-compatible robust scaling function
# using medians (center_) and IQR-based scales (scale_) computed offline.
# This mirrors sklearn's RobustScaler: (x - median) / IQR
def scaling(means, stds):
    def scalingActual(image):
        meanImg = ee.Image.constant(means)
        stdImg = ee.Image.constant(stds)

        return image.subtract(meanImg).divide(stdImg)
    return scalingActual

# Computes squared Euclidean distance between a pixel and a centroid
# Used to assign pixels to the nearest cluster in feature space.
def distCentroid(img, centroid):
    return img.subtract(centroid).pow(2).reduce(ee.Reducer.sum())

# Applies KMeans clustering inside Earth Engine using precomputed centroids
# and remaps cluster IDs to excavated / non-excavated labels.
def kmeansEE(centroids, clusterLabels):
    def kmeansEEActual(image):
        # Compute distance to each centroid
        distImgs = [distCentroid(image, ee.Image.constant(c)) for c in centroids]

        # Stack distances into a single image
        stack = ee.Image.cat(distImgs)

        # Earth Engine does not support argmin directly,
        # so we negate distances and use argmax instead
        negDistArray = stack.toArray().multiply(-1)
        clusterId = negDistArray.arrayArgmax().arrayGet([0])
        
        # Remap cluster IDs to excavated / non-excavated labels
        clusterIds = [int(k) for k in clusterLabels.keys()]
        excavatedStat = [clusterLabels[k] for k in clusterLabels.keys()]

        excavated = clusterId.remap(clusterIds, excavatedStat)

        # Preserve original mask and timestamp
        mask1 = image.mask().select(0)
        return excavated.updateMask(mask1).copyProperties(image, ["system:time_start"])
    return kmeansEEActual

# Simple morphological postprocessing to remove noise:
# opening followed by closing to smooth excavation masks.
def postprocessing(image):
    opened = image.focalMin(radius=1, units="pixels").focalMax(radius=1, units="pixels")
    closed = opened.focalMax(radius=1, units="pixels").focalMin(radius=1, units="pixels")
    return closed

# Extracts acquisition dates from exported EE band names
# and merges same-day images into a single mask.
def retrieveDates(bandNames, E):
    dates = []
    for name in bandNames:
        date = name.split("T")[0]
        dates.append(datetime.strptime(date, "%Y%m%d").date())

    merged_E = []
    merged_dates = []

    i = 0
    while i < len(dates):
        current_date = dates[i]
        day_imgs = [E[i]]
        j = i + 1

        # Collect all images from the same day
        while j < len(dates) and dates[j] == current_date:
            day_imgs.append(E[j])
            j += 1

        stack = np.stack(day_imgs)

        # If any pixel shows excavation → mark excavated
        has_excavation = np.any(stack == 1, axis=0)
        has_known = np.any(stack != 2, axis=0)

        # Default state = unknown (2)
        merged = np.full(stack.shape[1:], 2, dtype=np.uint8)
        merged[has_excavation] = 1
        merged[(~has_excavation) & has_known] = 0

        merged_E.append(merged)
        merged_dates.append(current_date)
        i = j

    merged_E = np.stack(merged_E)
    return merged_E, merged_dates

# Implements temporal confirmation logic:
# pixels must remain excavated for a threshold duration
# before being marked as confirmed.
def confidenceSystem(E, dates, threshold):
    T, H, W = E.shape

    confirmed = np.zeros((T, H, W), dtype=np.uint8)
    candidate = np.zeros((T, H, W), dtype=np.uint8)
    confidence = np.zeros((T, H, W), dtype=np.float32)

    daysExcavated = np.zeros((H, W), dtype=np.int32)
    firstSeen = np.full((H, W), -1, dtype=np.int32)
    confirmedFirstSeen = np.full((H, W), -1, dtype=np.int32)

    for t in range(T):
        Et = E[t].copy()
        deltaDays = 0 if t == 0 else (dates[t] - dates[t-1]).days
        prevConfirmed = np.zeros((H, W), dtype=np.uint8) if t == 0 else confirmed[t-1]

        working = (Et == 1) & (prevConfirmed == 0)
        reset = (Et == 0)
        newlySeen = (Et == 1) & (firstSeen == -1)
        unknown = (Et == 2)

        effectiveDelta = np.zeros((H, W), dtype=np.int32)
        effectiveDelta[~unknown] = deltaDays
        daysExcavated[working] += effectiveDelta[working]

        daysExcavated[reset] = 0
        firstSeen[reset] = -1
        firstSeen[newlySeen] = t

        newConfirmed = daysExcavated >= threshold
        confirmed[t] = prevConfirmed | newConfirmed

        # Record first confirmation time
        ys, xs = np.where(newConfirmed)
        confirmedFirstSeen[ys, xs] = firstSeen[ys, xs]

        # Candidate includes both confirmed and unconfirmed excavation
        candidate[t] = ((Et == 1) | (confirmed[t] == 1)).astype(np.uint8)

        confidence[t] = np.clip(daysExcavated / threshold, 0, 1)
        confidence[t][confirmed[t] == 1] = 1

    return confirmed, candidate, confidence, confirmedFirstSeen

# Retroactively marks excavation as confirmed
# from the first detection until confirmation.
def retroConfirm(confirmed, firstSeen):
    T, H, W = confirmed.shape
    retroConfirmed = np.zeros_like(confirmed)

    for t in range(T):
        retroConfirmed[t] = retroConfirmed[t-1] if t > 0 else 0
        newlyConfirmed = (confirmed[t] == 1) & (retroConfirmed[t] == 0)

        ys, xs = np.where(newlyConfirmed)
        for y, x in zip(ys, xs):
            t0 = firstSeen[y, x]
            if t0 >= 0:
                retroConfirmed[t0:t+1, y, x] = 1

    return retroConfirmed

# Loads the no-go zone polygon for a given mine if it exists.
def loadNogo(mineid, crs = None):
    path = f"./Mine Data/Mine_{mineid}_Data/nogozones.geojson"
    if not os.path.exists(path):
        return None

    nogo = gpd.read_file(path)
    if crs is not None:
        nogo = nogo.to_crs(crs)

    return nogo

# -------------------------------------------------------------------
# TRAINING PHASE (Part 1 - Feature Extraction in Earth Engine)
#
# This function:
# 1. Pulls Sentinel-2 imagery for a mine
# 2. Applies masking + feature engineering
# 3. Samples pixel-wise features inside the mine polygon
# 4. Exports the result as a CSV for offline clustering
#
# We intentionally start feature computation earlier than startTime
# to allow rolling statistics (medians, variance, slope) to stabilize.
# -------------------------------------------------------------------
def trainingStart(startTime, endTime, mineid, windowSize, debug = 0):
    startTime = ee.Date(startTime)

    # Extend the start time backwards so rolling windows have context
    actualStartTime = startTime.advance(-windowSize, "day")

    # Convert the mine polygon to an Earth Engine geometry
    mine = geemap.geopandas_to_ee(gdf[gdf["mine_id"] == mineid])

    # Load Sentinel-2 Surface Reflectance imagery
    # Filter spatially to the mine and temporally to the extended range
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(actualStartTime, endTime)
        .filterBounds(mine)
        .map(mask)
    )

    # Apply feature engineering (indices + rolling stats + slopes)
    # Then clip back to the actual training period
    s2 = featureEngineering(s2, windowSize).filterDate(startTime, endTime)

    # Convert images into a pixel-wise feature table
    # Each row corresponds to one pixel at one timestamp
    imageToTableUse = imageToTable(mine)
    s2 = s2.map(imageToTableUse).flatten().select(features)

    # Export the feature table to Google Drive for offline training
    task = ee.batch.Export.table.toDrive(
        collection = s2,
        description = f"mine_{mineid}_features_{debug}",
        fileFormat = "CSV"
    )
    task.start()

# -------------------------------------------------------------------
# TRAINING PHASE (Part 2 - Offline Clustering & Model Persistence)
#
# This function:
# 1. Loads the exported feature CSV
# 2. Applies RobustScaler to handle outliers
# 3. Performs KMeans clustering
# 4. Performs meta-clustering to separate excavated vs non-excavated
# 5. Prunes weak / false excavation clusters
# 6. Saves everything needed for monitoring
# -------------------------------------------------------------------
def trainingComplete(mineid, debug = 0, k = 6):
    mainDir = f"./Mine Data/Mine_{mineid}_data/"

    # Load extracted feature table
    df = pd.read_csv(mainDir + f"mine_{mineid}_features_{debug}.csv")
    df = df[features]

    # RobustScaler is used instead of StandardScaler because:
    # - Mining data contains heavy outliers
    # - We want median/IQR-based scaling
    scaler = RobustScaler(quantile_range=(25, 75))
    scaler.fit(df)

    # Numerical stabilization:
    # Very low variance in temporal features can cause scaling explosions
    slope_idx = features.index("NBR_slope")
    scaler.scale_[slope_idx] = max(scaler.scale_[slope_idx], 0.05)

    var_idx = features.index("NDVI_var")
    if scaler.scale_[var_idx] < 0.5:
        scaler.scale_[var_idx] = 1.0

    ndmi_var_idx = features.index("NDMI_var")
    if scaler.scale_[ndmi_var_idx] < 0.5:
        scaler.scale_[ndmi_var_idx] = 1.0

    # Scale the data and persist the scaler for monitoring
    dfScaled = scaler.transform(df)
    joblib.dump(scaler, mainDir + "scaler.pkl")

    # Primary clustering:
    # k is chosen empirically to separate distinct surface regimes
    kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 10)
    kmeans.fit(dfScaled)
    joblib.dump(kmeans, mainDir + "kmeans.pkl")

    # ----------------------------------------------------------------
    # META-CLUSTERING STEP
    #
    # Problem: Water and shadows can dominate clustering due to low SWIR
    # Solution: Clamp SWIR-like features before meta-clustering
    # ----------------------------------------------------------------
    centroids = kmeans.cluster_centers_
    centroids_for_meta = centroids.copy()

    swir_indices = [
        features.index("B11"),
        features.index("B12"),
        features.index("B11_median"),
        features.index("B12_median"),
        features.index("NDMI_median")
    ]

    for idx in swir_indices:
        centroids_for_meta[:, idx] = np.maximum(centroids_for_meta[:, idx], -1.0)

    # Meta-clustering separates excavated vs non-excavated clusters
    metaKMeans = KMeans(n_clusters = 2, random_state = 42, n_init = 10)
    metaLabels = metaKMeans.fit_predict(centroids_for_meta)

    # Identify which metacluster corresponds to excavation
    exLabel, means = findExcavated(centroids, metaLabels)

    # ----------------------------------------------------------------
    # CLUSTER PRUNING
    #
    # Goal: Retain strong excavation clusters, discard weak / noisy ones
    # Strategy:
    # - Identify the strongest excavation cluster
    # - Keep secondary clusters only if they are sufficiently similar
    # ----------------------------------------------------------------
    b12_idx = features.index("B12_median")
    bsi_idx = features.index("BSI_median")
    ndvi_idx = features.index("NDVI_median")
    ndmi_idx = features.index("NDMI_median")

    valid_indices = [i for i in range(k) if metaLabels[i] == exLabel]
    cluster_scores = {}

    for i in valid_indices:
        score = (
            2.0 * centroids[i][b12_idx]
            + 1.0 * centroids[i][bsi_idx]
            - 0.5 * centroids[i][ndvi_idx]
            - 1.0 * centroids[i][ndmi_idx]
        )
        cluster_scores[i] = score

    leader_idx = max(cluster_scores, key=cluster_scores.get)
    leader_score = cluster_scores[leader_idx]

    final_labels = []

    for i in range(k):
        is_mine = False

        if metaLabels[i] == exLabel:
            current_score = cluster_scores.get(i, -np.inf)

            # Always keep the strongest excavation cluster
            if i == leader_idx:
                is_mine = True
            # Keep secondary clusters only if sufficiently strong
            elif current_score > (leader_score * 0.60):
                is_mine = True

            # Hard safety filters
            if current_score < 0.5:
                is_mine = False
            if centroids[i][ndvi_idx] > 0.5:
                is_mine = False

        final_labels.append(1 if is_mine else 0)

    # Persist cluster labels and centroids for monitoring
    metadata = {
        "cluster_labels": {str(i): final_labels[i] for i in range(k)},
        "centroids": {
            str(i): {
                features[j]: float(centroids[i][j])
                for j in range(len(features))
            }
            for i in range(k)
        },
        "excavated_metacluster": int(exLabel),
        "metacluster_stats": {
            "0": {"excavation_score": float(means[0])},
            "1": {"excavation_score": float(means[1])}
        }
    }

    with open(mainDir + "clusterData.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
# -------------------------------------------------------------------
# MONITORING PHASE (Part 1 - Inference in Earth Engine)
#
# This function:
# 1. Loads Sentinel-2 imagery for a monitoring period
# 2. Re-applies the same feature engineering as training
# 3. Applies RobustScaler and KMeans *inside Earth Engine*
# 4. Produces a binary excavation mask per timestamp
# 5. Exports the full temporal stack as a multi-band GeoTIFF
#
# Values:
#   1 -> Excavated
#   0 -> Non-excavated
#   2 -> Unknown / masked
# -------------------------------------------------------------------
def monitoringStart(startTime, endTime, mineid, windowSize, debug = 0):
    mainDir = f"./Mine Data/Mine_{mineid}_data/"

    startTime = ee.Date(startTime)
    actualStartTime = startTime.advance(-windowSize, "day")

    # Convert mine polygon to Earth Engine geometry
    mine = geemap.geopandas_to_ee(gdf[gdf["mine_id"] == mineid])

    # Load and preprocess Sentinel-2 imagery
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(actualStartTime, endTime)
        .filterBounds(mine)
        .map(mask)
    )

    # Apply identical feature engineering as training
    s2 = featureEngineering(s2, windowSize).filterDate(startTime, endTime)

    # ----------------------------------------------------------------
    # MANUAL SCALING IN EARTH ENGINE
    #
    # Earth Engine does not support sklearn,
    # so we manually re-implement RobustScaler using
    # the learned center_ and scale_ parameters.
    # ----------------------------------------------------------------
    scaler = joblib.load(mainDir + "scaler.pkl")
    means = scaler.center_.tolist()
    stds = scaler.scale_.copy()
    stds[stds == 0] = 1.0
    stds = stds.tolist()

    eeScaling = scaling(means, stds)
    s2 = s2.map(eeScaling)

    # ----------------------------------------------------------------
    # MANUAL KMEANS ASSIGNMENT IN EARTH ENGINE
    #
    # We compute distance to each centroid explicitly
    # and assign the nearest cluster.
    # ----------------------------------------------------------------
    kmeans = joblib.load(mainDir + "kmeans.pkl")
    centroids = kmeans.cluster_centers_.tolist()

    with open(mainDir + "clusterData.json") as f:
        clusterData = json.load(f)

    clusterLabels = clusterData["cluster_labels"]
    exMaskCreate = kmeansEE(centroids, clusterLabels)
    s2 = s2.map(exMaskCreate)

    # Morphological cleanup:
    # Opening removes isolated false positives
    # Closing fills small holes inside excavation regions
    s2 = s2.map(postprocessing)

    # Ensure temporal ordering
    s2 = s2.sort("system:time_start")

    # Assign value 2 to masked pixels (unknown state)
    s2 = s2.map(lambda img: img.unmask(2))

    # Convert ImageCollection -> single Image with one band per date
    # This preserves full temporal information in one GeoTIFF
    s2 = s2.toBands().clip(mine)

    # Export the result for offline analysis
    task = ee.batch.Export.image.toDrive(
        image = s2,
        description = f"mine_{mineid}_excavation_{debug}",
        fileNamePrefix = f"mine_{mineid}_excavation_{debug}",
        region = mine.geometry(),
        scale = 10,
        maxPixels = 1e13
    )
    task.start()

# -------------------------------------------------------------------
# MONITORING PHASE (Part 2 - Offline Temporal Reasoning)
#
# This function:
# 1. Loads exported GeoTIFF(s)
# 2. Merges overlapping tiles per day
# 3. Builds a temporal confidence system
# 4. Produces final excavation timelines and visualizations
# 5. Generates no-go zone alerts if applicable
# -------------------------------------------------------------------
def monitoringComplete(mineid, threshold, debug = 0):
    mainDir = f"./Mine Data/Mine_{mineid}_data/"
    outDir = f"./Mine Data/Mine_{mineid}_data/Outputs/"

    alert_log_path = os.path.join(outDir, f"mine_{mineid}_alerts.log")

    # Allow processing multiple GeoTIFF chunks if needed
    if isinstance(debug, (int, str)):
        debug_list = [debug]
    else:
        debug_list = list(debug)

    all_E = []
    all_dates = []
    transform = None
    crs = None

    # ----------------------------------------------------------------
    # LOAD AND MERGE RASTERS
    #
    # Each band corresponds to a timestamp,
    # but overlapping Sentinel tiles can produce
    # multiple bands for the same day.
    # ----------------------------------------------------------------
    for d in debug_list:
        tif_path = mainDir + f"mine_{mineid}_excavation_{d}.tif"

        with rasterio.open(tif_path) as src:
            E = src.read()
            bandNames = src.descriptions

            if transform is None:
                transform = src.transform
                crs = src.crs

        # Merge multiple bands from the same date
        Efixed, dates = retrieveDates(bandNames, E)

        all_E.append(Efixed)
        all_dates.extend(dates)

    # Load no-go zone polygon if present
    nogo = loadNogo(mineid, crs)

    if nogo is not None:
        with open(alert_log_path, "w") as f:
            f.write(f"# Monitoring run started: {datetime.now()}\n")

    # Stack all temporal data
    Efixed = np.concatenate(all_E, axis=0)

    # Sort by time
    sort_idx = np.argsort(all_dates)
    Efixed = Efixed[sort_idx]
    dates = [all_dates[i] for i in sort_idx]

    # ----------------------------------------------------------------
    # CONFIDENCE SYSTEM
    #
    # Excavation must persist over time.
    # Single-frame changes are treated as noise.
    # ----------------------------------------------------------------
    confirmed, candidate, confidence, firstSeen = confidenceSystem(
        Efixed, dates, threshold
    )

    # Retro-confirmation removes latency bias:
    # pixels confirmed later are marked as excavated
    # starting from their first observed excavation.
    retroConfirmed = retroConfirm(confirmed, firstSeen)

    # ----------------------------------------------------------------
    # OUTPUT GENERATION
    #
    # Spatial maps, temporal plots, and summaries
    # ----------------------------------------------------------------
    makeSpatialMaps(
        mineid, outDir, dates,
        retroConfirmed, confidence,
        transform, crs, nogo
    )

    retroConfirmedAnalysis(
        mineid, dates, outDir,
        retroConfirmed, transform, crs, nogo
    )

    retroAreas = ExcavationTimePlot(
        mineid, outDir,
        retroConfirmed, dates, transform
    )

    candAreas = CandidateExcavationTimePlot(
        mineid, outDir,
        candidate, dates, transform
    )

    growthRate = GrowthRatePlot(
        mineid, outDir,
        retroConfirmed, dates, transform
    )

    ComparisionPlot(
        mineid, outDir,
        retroAreas, candAreas
    )

    NormalizedExcavationPlot(
        mineid, outDir,
        retroAreas
    )

    FirstSeenDateMap(
        mineid, outDir,
        firstSeen, dates,
        transform, crs, nogo
    )

    # ----------------------------------------------------------------
    # NO-GO ZONE ALERT SYSTEM
    #
    # Alerts are hierarchical:
    # LEVEL 1 -> candidate intrusion
    # LEVEL 2 -> confirmed violation
    # LEVEL 3 -> sustained expansion
    # ----------------------------------------------------------------
    NoGoAlertSystem(
        mineid, outDir,
        dates, candidate, confirmed,
        transform, crs, nogo
    )

    NoGoExcavationTimePlot(
        mineid, outDir,
        dates, candidate, confirmed,
        transform, crs, nogo
    )

    # Debug snapshots at representative times
    for i in [
        int(round(p/100 * (len(dates)-1)))
        for p in [0, 20, 40, 60, 80, 100]
    ]:
        debugCluster(mineid, i, dates, Efixed)