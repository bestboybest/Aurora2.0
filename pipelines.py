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
    # At the end of findExcavated(), before the return, add:
    if scores[0] < 0 and scores[1] < 0:
    # Both negative — pick relatively better one, shift doesn't matter
        pass  # existing logic already picks the higher one correctly
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

# -------------------------------------------------------------------
# SAR SECTION
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# SAR Monitoring (Part 1 - Earth Engine Export)
#
# Exports a binary excavation mask derived from Sentinel-1 GRD
# backscatter. Values: 1=excavated, 0=non-excavated, 2=unknown/masked
#
# This is intentionally simple — SAR is used only as a gap-filler
# for cloud-covered periods, not as the primary classification.
# A speckle filter is applied first to reduce noise.
# made more conservative by requiring agreement between absolute and seasonal thresholds.
# -------------------------------------------------------------------
def sarMonitoringStart(startTime, endTime, mineid, debug=0, baseline_days=60):
    
    import rasterio
    import numpy as np

    mine    = geemap.geopandas_to_ee(gdf[gdf["mine_id"] == mineid])
    mainDir = f"./Mine Data/Mine_{mineid}_data/"

    # ------------------------------------------------------------------
    # STEP 1 — Derive mine-specific thresholds from S2 confirmed labels
    # ------------------------------------------------------------------
    # Defaults (same as the old hard-coded rule)
    vv_thresh_val    = -12.0   # absolute VV threshold (dB)
    delta_thresh_val =   2.0   # delta VV threshold (dB above baseline)

    tif_path = mainDir + f"mine_{mineid}_excavation_{debug}.tif"
    if os.path.exists(tif_path):
        with rasterio.open(tif_path) as src:
            E         = src.read()          # (T, H, W)  values 0/1/2
            transform = src.transform
            crs       = src.crs

        # Aggregate across time: a pixel is "confirmed excavated" if it
        # ever shows 1, "confirmed clear" if it ever shows 0 and never 1.
        ever_excavated = np.any(E == 1, axis=0)
        ever_clear     = np.any(E == 0, axis=0) & ~ever_excavated

        # Sample pixel centres for each class (cap at 1000 each)
        
        def pixel_geometries(mask_2d, transform, crs, cap=1000):
            rows, cols = np.where(mask_2d)
            if len(rows) > cap:
                idx  = np.random.choice(len(rows), cap, replace=False)
                rows, cols = rows[idx], cols[idx]
            import rasterio.transform as rio_t
            lons, lats = rio_t.xy(transform, rows, cols, offset="center")
            
            # Extract the EPSG string or WKT string from the rasterio CRS object
            crs_string = crs.to_string() if hasattr(crs, 'to_string') else "EPSG:4326"
            
            return [
                ee.Feature(ee.Geometry.Point([float(lo), float(la)], crs_string))
                for lo, la in zip(lons, lats)
            ]

        ex_pts  = pixel_geometries(ever_excavated, transform, crs)
        clr_pts = pixel_geometries(ever_clear,     transform, crs)

        if len(ex_pts) > 0:
            # Pull a recent SAR image to sample VV at the labelled pixels
            # (use the middle of the monitoring period for stability)
            mid_date = ee.Date(startTime).advance(
                ee.Date(endTime).difference(ee.Date(startTime), "day").divide(2), "day"
            )
            ref_sar = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterDate(
                    mid_date.advance(-30, "day"),
                    mid_date.advance( 30, "day")
                )
                .filterBounds(mine)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .select(["VV"])
                .map(lambda img: img.focalMean(3, "square", "pixels"))
                .median()
            )

            def sample_vv(pts):
                fc      = ee.FeatureCollection(pts)
                sampled = ref_sar.sampleRegions(collection=fc, scale=10, geometries=False)
                vals    = [
                    f["properties"].get("VV")
                    for f in sampled.getInfo()["features"]
                    if f["properties"].get("VV") is not None
                ]
                return np.array(vals, dtype=np.float32)

            vv_ex  = sample_vv(ex_pts)
            vv_clr = sample_vv(clr_pts) if len(clr_pts) > 0 else np.array([])

            if len(vv_ex) >= 10:
                # Absolute threshold: lower edge of excavated distribution
                # (mean - 1σ) so we don't miss pixels that are slightly
                # noisier than the centre of the excavated cluster.
                vv_thresh_val = float(np.mean(vv_ex) - np.std(vv_ex))

                # Delta threshold: how much brighter excavated pixels are
                # compared to non-excavated. Use half the separation as
                # the minimum delta required.
                if len(vv_clr) >= 10:
                    separation    = float(np.mean(vv_ex) - np.mean(vv_clr))
                    delta_thresh_val = max(1.0, separation * 0.5)

                print(
                    f"Mine {mineid} — derived thresholds from {len(vv_ex)} excavated "
                    f"/ {len(vv_clr)} clear pixels:\n"
                    f"  vv_thresh    = {vv_thresh_val:.2f} dB  (was -12 dB)\n"
                    f"  delta_thresh = {delta_thresh_val:.2f} dB  (was fixed +2 dB)"
                )
            else:
                print(f"Mine {mineid} — too few labelled pixels ({len(vv_ex)}); "
                      f"using fallback thresholds (-12 dB / +2 dB)")
        else:
            print(f"Mine {mineid} — no excavated pixels found in S2 GeoTIFF; "
                  f"using fallback thresholds")
    else:
        print(f"Mine {mineid} — S2 GeoTIFF not found at {tif_path}; "
              f"using fallback thresholds (-12 dB / +2 dB)")

    # ------------------------------------------------------------------
    # STEP 2 — Build the full SAR collection (extended back for baseline)
    # ------------------------------------------------------------------
    # Pull `baseline_days` extra history so the very first images in the
    # monitoring window have a full rolling baseline to compare against.
    actualStartTime = ee.Date(startTime).advance(-baseline_days, "day")

    sar_full = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(actualStartTime, endTime)
        .filterBounds(mine)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
        .map(lambda img: ee.Image.cat([
            img.select("VV").focalMean(3, "square", "pixels").rename("VV"),
            img.select("VH").focalMean(3, "square", "pixels").rename("VH"),
        ]).copyProperties(img, ["system:time_start"]))
    )

    # ------------------------------------------------------------------
    # STEP 3 — classifySAR: mine-adaptive + season-adaptive thresholding
    # ------------------------------------------------------------------
    vv_thresh_ee    = ee.Number(vv_thresh_val)
    delta_thresh_ee = ee.Number(delta_thresh_val)

    def classifySAR(image):
        date = ee.Date(image.get("system:time_start"))

        # ---- Absolute signal ----------------------------------------
        vv = image.select("VV")
        vh = image.select("VH")

        # Mine-specific absolute threshold
        absolute_flag = vv.gt(vv_thresh_ee)  # VV brighter than mine baseline

        # ---- Seasonal / rolling baseline signal ---------------------
        # Pixel-wise median of all SAR passes in the preceding window.
        # Using the full extended collection so baseline is always available.
        baseline = (
            sar_full
            .filterDate(date.advance(-baseline_days, "day"), date)
            .select("VV")
            .median()                           # robust to speckle outliers
        )

        delta_vv     = vv.subtract(baseline)   # positive = brighter than season
        delta_flag   = delta_vv.gt(delta_thresh_ee)

        # ---- Combined rule (both must agree) ------------------------
        # This preserves the same conservatism as the old two-consecutive-
        # pass agreement rule: a single noisy image cannot trigger excavation.
        excavated = absolute_flag.And(delta_flag).rename("excavated")

        return (
            excavated
            .unmask(2)
            .toByte()
            .copyProperties(image, ["system:time_start"])
        )

    # Only export the monitoring window (not the baseline warm-up period)
    sar = (
        sar_full
        .filterDate(startTime, endTime)
        .map(classifySAR)
        .sort("system:time_start")
    )
    
    reference = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(mine)
    .first()
    .select("B2")
    )

    proj = reference.projection()

    sar = (
        sar
        .toBands()
        .reproject(
            crs=proj.crs(),
            scale=10
        )
        .clip(mine)
    )

    task = ee.batch.Export.image.toDrive(
        image=sar,
        description=f"mine_{mineid}_sar_{debug}",
        fileNamePrefix=f"mine_{mineid}_sar_{debug}",
        region=mine.geometry(),
        scale=10,
        maxPixels=1e13
    )
    task.start()

# -------------------------------------------------------------------
# SAR Monitoring (Part 2 - Load SAR GeoTIFF into numpy)
#
# Loads the exported SAR GeoTIFF and returns (sar_E, sar_dates)
# in the same format as Efixed — shape (T, H, W), values 0/1/2.
#
# S1 band names from EE look like:
#   S1A_IW_GRDH_1SDV_20220103T...
# The date is extracted by finding the first 8-digit numeric token.
# -------------------------------------------------------------------
def parseSARDates(bandNames):
    import re
    dates = []
    for name in bandNames:
        # Find first 8-digit sequence in the band name — that's YYYYMMDD
        match = re.search(r'(\d{8})', name)
        if match:
            dates.append(datetime.strptime(match.group(1), "%Y%m%d").date())
        else:
            raise ValueError(f"Could not parse date from SAR band name: {name}")
    return dates

def loadSAR(mineid, debug=0):
    mainDir  = f"./Mine Data/Mine_{mineid}_data/"
    tif_path = mainDir + f"mine_{mineid}_sar_{debug}.tif"

    if not os.path.exists(tif_path):
        return None, None

    with rasterio.open(tif_path) as src:
        E         = src.read()
        bandNames = src.descriptions

    # Parse SAR-specific band name format (S1A_IW_GRDH_1SDV_YYYYMMDD...)
    dates = parseSARDates(bandNames)

    # Merge same-day observations (same logic as S2)
    merged_E     = []
    merged_dates = []

    i = 0
    while i < len(dates):
        current_date = dates[i]
        day_imgs     = [E[i]]
        j = i + 1

        while j < len(dates) and dates[j] == current_date:
            day_imgs.append(E[j])
            j += 1

        stack         = np.stack(day_imgs)
        has_excavation = np.any(stack == 1, axis=0)
        has_known      = np.any(stack != 2, axis=0)

        merged        = np.full(stack.shape[1:], 2, dtype=np.uint8)
        merged[has_excavation] = 1
        merged[(~has_excavation) & has_known] = 0

        merged_E.append(merged)
        merged_dates.append(current_date)
        i = j

    return np.stack(merged_E), merged_dates

# -------------------------------------------------------------------
# SAR Gap-Fill
#
# Fills unknown (2) pixels in Efixed using SAR classifications,
# but ONLY where S2 had no valid observation on that date.
#
# consecutive_required: how many nearby SAR observations must agree
#   before filling. Default=2 (~12 days of consistent SAR signal).
#   Increase to 3 if candidate area inflates during monsoon.
#
# max_gap_days: SAR observations within this window of the S2 date
#   are considered "nearby". Matches S1 revisit (~6 days).
#
# Fill logic:
#   - Excavated fill: ALL nearby SAR obs agree = 1
#   - Clear fill:     ALL nearby SAR obs agree = 0
#   - Disagreement:   pixel stays 2 (unknown)
#
# This means a single noisy SAR image cannot confirm excavation.
# Both passes must agree, giving ~12 days of consistent signal
# as the minimum bar — appropriate for persistent surface change.
# -------------------------------------------------------------------
def sarGapFill(Efixed, dates, sar_E, sar_dates, consecutive_required=2, max_gap_days=6):
    if sar_E is None or sar_dates is None:
        return Efixed

    filled = Efixed.copy()

    for t, date in enumerate(dates):
        unknown_mask = Efixed[t] == 2

        # If S2 is fully valid on this date, skip — SAR not needed
        if not np.any(unknown_mask):
            continue

        # Find ALL SAR observations within the time window
        nearby_indices = [
            i for i, sd in enumerate(sar_dates)
            if abs((date - sd).days) <= max_gap_days
        ]

        # Not enough SAR passes nearby — leave as unknown
        if len(nearby_indices) < consecutive_required:
            continue

        # Stack all nearby SAR slices: shape (N, H, W)
        nearby_slices = np.stack([sar_E[i] for i in nearby_indices])

        # Consensus check: ALL nearby observations must agree
        sar_agrees_excavated = np.all(nearby_slices == 1, axis=0)
        sar_agrees_clear     = np.all(nearby_slices == 0, axis=0)

        # Only fill pixels that are unknown in S2
        fillable_excavated = unknown_mask & sar_agrees_excavated
        fillable_clear     = unknown_mask & sar_agrees_clear

        filled[t][fillable_excavated] = 1
        filled[t][fillable_clear]     = 0
        # Pixels where SAR disagrees stay as 2 (unknown)

    return filled

# -------------------------------------------------------------------
# Implements temporal confirmation logic:
# pixels must remain excavated for a threshold duration
# before being marked as confirmed.
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Retroactively marks excavation as confirmed
# from the first detection until confirmation.
# -------------------------------------------------------------------
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
def loadNogo(mineid, crs=None):
    path = f"./Mine Data/Mine_{mineid}_Data/nogozones.geojson"
    if not os.path.exists(path):
        return None

    nogo = gpd.read_file(path)
    if crs is not None:
        nogo = nogo.to_crs(crs)

    return nogo


# -------------------------------------------------------------------
# SAR ANALYSIS — Phase 1: Export SAR values for labelled S2 pixels
#
# This function is the FIRST STEP before building any dynamic SAR
# threshold. It lets you verify whether SAR signals (VV, VH, VV/VH)
# actually differ between excavated and non-excavated pixels.
#
# What it does:
#   1. Loads the S2 excavation GeoTIFF already produced by
#      monitoringStart() (values: 1=excavated, 0=non-excavated, 2=masked)
#   2. For each S2 date, finds the closest Sentinel-1 pass (within
#      max_sar_gap_days) for that mine
#   3. Samples VV and VH values from those SAR images at pixels that
#      are labelled 1 or 0 in S2
#   4. Exports a CSV with columns:
#        date, pixel_label, VV, VH, VV_VH_ratio
#      where pixel_label is:
#        "excavated"    → S2 said 1
#        "non-excavated"→ S2 said 0
#        "cloud-masked" → S2 said 2 (SAR values recorded anyway)
#
# Usage:
#   sarAnalysisStart("2023-01-01", "2023-12-31", mineid=0, s2_debug=0)
#   # Wait for the EE task to complete, then:
#   sarAnalysisPlot(mineid=0)
#
# Parameters:
#   startTime / endTime : monitoring period (same as monitoringStart)
#   mineid              : mine index
#   s2_debug            : the debug suffix used in monitoringStart()
#                         so we load the right GeoTIFF
#   max_sar_gap_days    : SAR pass is used only if within this many
#                         days of the S2 date (default 6 = S1 revisit)
#   max_pixels_per_class: cap on how many pixels to sample per class
#                         per date to keep CSV size manageable
#   debug               : suffix for the output CSV filename
# -------------------------------------------------------------------
def sarAnalysisStart(
    startTime, endTime, mineid,
    s2_debug=0,
    max_sar_gap_days=6,
    max_pixels_per_class=500,
    debug=0
):
    import rasterio
    import numpy as np
    import random

    mainDir = f"./Mine Data/Mine_{mineid}_data/"
    tif_path = mainDir + f"mine_{mineid}_excavation_{s2_debug}.tif"

    mine = geemap.geopandas_to_ee(gdf[gdf["mine_id"] == mineid])

    # ------------------------------------------------------------------
    # Load the S2 labels (GeoTIFF produced by monitoringStart)
    # ------------------------------------------------------------------
    with rasterio.open(tif_path) as src:
        E         = src.read()           # shape: (T, H, W)
        bandNames = list(src.descriptions)
        transform = src.transform
        crs_wkt   = src.crs.to_wkt()

    # Parse dates from band names (same format as retrieveDates)
    s2_dates = []
    for name in bandNames:
        date_str = name.split("T")[0]
        s2_dates.append(datetime.strptime(date_str, "%Y%m%d").date())

    # ------------------------------------------------------------------
    # Load ALL Sentinel-1 imagery for the mine in the date range
    # We keep VV and VH, apply the same speckle filter as classifySAR
    # ------------------------------------------------------------------
    sar = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(startTime, endTime)
        .filterBounds(mine)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )

    def smoothSAR(image):
        vv = image.select("VV").focalMean(3, "square", "pixels").rename("VV")
        vh = image.select("VH").focalMean(3, "square", "pixels").rename("VH")
        ratio = vv.subtract(vh).rename("VV_VH_ratio")   # dB difference ≡ log ratio
        return (
            ee.Image.cat([vv, vh, ratio])
            .copyProperties(image, ["system:time_start"])
        )

    sar = sar.map(smoothSAR)

    # Retrieve SAR dates from EE (we need to match them to S2 dates client-side)
    sar_dates_info = sar.aggregate_array("system:time_start").getInfo()
    sar_dates = [
        datetime.utcfromtimestamp(ms / 1000).date()
        for ms in sar_dates_info
    ]
    sar_list = sar.toList(sar.size())

    # ------------------------------------------------------------------
    # Helper: convert raster (row, col) → lon/lat using the GeoTIFF
    # transform so we can create EE point samples
    # ------------------------------------------------------------------
    import rasterio.transform as rio_transform

    def rc_to_lonlat(row, col, transform):
        lon, lat = rio_transform.xy(transform, row, col, offset="center")
        return lon, lat

    # ------------------------------------------------------------------
    # For each S2 date, find the nearest SAR pass and sample pixels
    # ------------------------------------------------------------------
    all_records = []

    for t, s2_date in enumerate(s2_dates):
        label_map = E[t]   # (H, W), values 0/1/2

        # Find closest SAR pass
        closest_sar_idx = None
        closest_gap     = 9999
        for i, sd in enumerate(sar_dates):
            gap = abs((s2_date - sd).days)
            if gap <= max_sar_gap_days and gap < closest_gap:
                closest_gap     = gap
                closest_sar_idx = i

        if closest_sar_idx is None:
            # No SAR pass close enough — skip this S2 date
            continue

        sar_image = ee.Image(sar_list.get(closest_sar_idx))

        # Collect pixel positions for each class
        rows_ex,  cols_ex  = np.where(label_map == 1)
        rows_non, cols_non = np.where(label_map == 0)
        rows_cld, cols_cld = np.where(label_map == 2)

        # Cap to avoid GEE quota issues
        def sample_indices(rows, cols, cap):
            idx = list(range(len(rows)))
            if len(idx) > cap:
                idx = random.sample(idx, cap)
            return rows[idx], cols[idx]

        rows_ex,  cols_ex  = sample_indices(rows_ex,  cols_ex,  max_pixels_per_class)
        rows_non, cols_non = sample_indices(rows_non, cols_non, max_pixels_per_class)
        rows_cld, cols_cld = sample_indices(rows_cld, cols_cld, max_pixels_per_class // 4)

        class_pixels = [
            ("excavated",     rows_ex,  cols_ex),
            ("non-excavated", rows_non, cols_non),
            ("cloud-masked",  rows_cld, cols_cld),
        ]

        for label_name, rows, cols in class_pixels:
            if len(rows) == 0:
                continue

            # Build an EE FeatureCollection of point geometries
            points = [
                ee.Feature(
                    ee.Geometry.Point(rc_to_lonlat(r, c, transform))
                )
                for r, c in zip(rows.tolist(), cols.tolist())
            ]
            fc = ee.FeatureCollection(points)

            # Sample SAR values at those points
            sampled = sar_image.sampleRegions(
                collection=fc,
                scale=10,
                geometries=False
            )

            records = sampled.getInfo()["features"]
            for rec in records:
                props = rec["properties"]
                all_records.append({
                    "date":         s2_date.isoformat(),
                    "pixel_label":  label_name,
                    "VV":           props.get("VV"),
                    "VH":           props.get("VH"),
                    "VV_VH_ratio":  props.get("VV_VH_ratio"),
                })

        print(f"  [{t+1}/{len(s2_dates)}] {s2_date} — SAR pass {sar_dates[closest_sar_idx]} "
              f"(gap {closest_gap}d) — "
              f"ex:{len(rows_ex)} non:{len(rows_non)} cld:{len(rows_cld)}")

    # ------------------------------------------------------------------
    # Save the CSV
    # ------------------------------------------------------------------
    out_csv = mainDir + f"mine_{mineid}_sar_analysis_{debug}.csv"
    df_out  = pd.DataFrame(all_records)
    df_out.dropna(subset=["VV", "VH"], inplace=True)
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df_out)} pixel records → {out_csv}")
    return df_out


# -------------------------------------------------------------------
# SAR ANALYSIS — Phase 2: Histogram plots
#
# Loads the CSV produced by sarAnalysisStart() and generates 3 × 3
# overlaid histograms:
#
#   Rows    → feature:  VV  |  VH  |  VV/VH ratio
#   Columns → class:    excavated (red)
#                       non-excavated (green)
#                       cloud-masked (grey)
#
# Each subplot also prints the class medians so you can eyeball
# whether a fixed threshold (e.g. VV > -12 dB) is reasonable or
# whether mine-specific stats would do better.
#
# The figure is saved to:
#   ./Mine Data/Mine_{mineid}_data/Outputs/mine_{mineid}_sar_histograms_{debug}.png
#
# Usage:
#   sarAnalysisPlot(mineid=0)         # loads sar_analysis_0.csv
#   sarAnalysisPlot(mineid=0, debug=1)
# -------------------------------------------------------------------
def sarAnalysisPlot(mineid, debug=0):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    mainDir = f"./Mine Data/Mine_{mineid}_data/"
    outDir  = f"./Mine Data/Mine_{mineid}_data/Outputs/"
    os.makedirs(outDir, exist_ok=True)

    csv_path = mainDir + f"mine_{mineid}_sar_analysis_{debug}.csv"
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["VV", "VH", "VV_VH_ratio"])

    features_sar = ["VV", "VH", "VV_VH_ratio"]
    labels       = ["excavated", "non-excavated", "cloud-masked"]
    colors       = {"excavated": "#d62728", "non-excavated": "#2ca02c", "cloud-masked": "#7f7f7f"}
    xlabels      = {
        "VV":          "VV backscatter (dB)",
        "VH":          "VH backscatter (dB)",
        "VV_VH_ratio": "VV − VH (dB)",
    }

    fig, axes = plt.subplots(
        nrows=len(features_sar), ncols=len(labels),
        figsize=(14, 10),
        sharey=False
    )
    fig.suptitle(
        f"Mine {mineid} — SAR feature distributions by S2 pixel label\n"
        f"(red=excavated  green=non-excavated  grey=cloud-masked)",
        fontsize=13, y=1.01
    )

    for row_idx, feat in enumerate(features_sar):
        for col_idx, label in enumerate(labels):
            ax  = axes[row_idx][col_idx]
            sub = df[df["pixel_label"] == label][feat].dropna()

            if len(sub) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
                ax.set_title(f"{label}\n{feat}", fontsize=9)
                continue

            median_val = sub.median()
            color      = colors[label]

            ax.hist(sub, bins=60, color=color, alpha=0.75, edgecolor="none")
            ax.axvline(median_val, color="black", linewidth=1.5,
                       linestyle="--", label=f"median {median_val:.2f}")

            # Mark the current fixed threshold on VV plots for reference
            if feat == "VV":
                ax.axvline(-12, color="orange", linewidth=1.2,
                           linestyle=":", label="current threshold (−12 dB)")

            ax.set_xlabel(xlabels[feat], fontsize=8)
            ax.set_ylabel("pixel count",  fontsize=8)
            ax.set_title(f"{label}  (n={len(sub):,})\nmedian {feat} = {median_val:.2f}",
                         fontsize=9)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    out_png = outDir + f"mine_{mineid}_sar_histograms_{debug}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved histograms → {out_png}")

    # ------------------------------------------------------------------
    # Print a concise summary table so you can read the numbers directly
    # ------------------------------------------------------------------
    print(f"\n{'Feature':<14} {'Class':<18} {'Median':>8} {'Mean':>8} {'Std':>8} {'N':>7}")
    print("-" * 60)
    for feat in features_sar:
        for label in labels:
            sub = df[df["pixel_label"] == label][feat].dropna()
            if len(sub) == 0:
                continue
            print(f"{feat:<14} {label:<18} {sub.median():>8.2f} {sub.mean():>8.2f} "
                  f"{sub.std():>8.2f} {len(sub):>7,}")
        print()

    return df

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
def trainingStart(startTime, endTime, mineid, windowSize, debug=0):
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
        collection=s2,
        description=f"mine_{mineid}_features_{debug}",
        fileFormat="CSV"
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
def trainingComplete(mineid, debug=0, k=8):
    mainDir = f"./Mine Data/Mine_{mineid}_data/"
    outDir  = f"./Mine Data/Mine_{mineid}_data/Outputs/"

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
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(dfScaled)
    joblib.dump(kmeans, mainDir + "kmeans.pkl")
    labels = kmeans.labels_

    # ----------------------------------------------------------------
    # META-CLUSTERING STEP
    #
    # Problem: Water and shadows can dominate clustering due to low SWIR
    # Solution: Clamp SWIR-like features before meta-clustering
    # ----------------------------------------------------------------
    centroids = kmeans.cluster_centers_

    dfSuppressed = dfScaled.copy()

    low_indices = [
        features.index("B11"),
        features.index("B12"),
        features.index("B11_median"),
        features.index("B12_median"),
        features.index("BSI_median")
    ]

    high_indices = [
        features.index("NDVI"),
        features.index("NDVI_median"),
        features.index("NDMI_median"),
        features.index("NBR_slope")
    ]

    # Hard clamp at (q1 - 1.5*iqr) and (q3 + 1.5*iqr) to catch
    # true outliers without being overly aggressive
    for idx in low_indices:
        q3 = np.percentile(dfScaled[:, idx], 75)
        q1 = np.percentile(dfScaled[:, idx], 25)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        dfSuppressed[:, idx] = np.maximum(dfSuppressed[:, idx], lower_bound)
    
    for idx in high_indices:
        q3 = np.percentile(dfScaled[:, idx], 75)
        q1 = np.percentile(dfScaled[:, idx], 25)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        dfSuppressed[:, idx] = np.minimum(dfSuppressed[:, idx], upper_bound) 
    
    centroids_for_meta = np.zeros_like(centroids)

    for i in range(k):
        cluster_pixels = dfSuppressed[labels == i]
        if len(cluster_pixels) == 0:
            centroids_for_meta[i] = centroids[i]  # fallback
        else:
            centroids_for_meta[i] = np.mean(cluster_pixels, axis=0)

    # Meta-clustering separates excavated vs non-excavated clusters
    metaKMeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    metaLabels = metaKMeans.fit_predict(centroids_for_meta)

    # Identify which metacluster corresponds to excavation
    exLabel, means = findExcavated(centroids, metaLabels)

    # ----------------------------------------------------------------
    # CLUSTER PRUNING
    # ----------------------------------------------------------------
    b12_idx  = features.index("B12_median")
    bsi_idx  = features.index("BSI_median")
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
    
    # After the loop, shift all scores so the best is always positive
    if cluster_scores:
        min_score = min(cluster_scores.values())
        if min_score < 0:
            shift = abs(min_score) + 0.1
            cluster_scores = {i: s + shift for i, s in cluster_scores.items()}

    if len(cluster_scores) == 0:
        selected_clusters = set()
    else:
        selected_clusters = set()

        scores       = list(cluster_scores.values())
        best_score   = max(scores)
        distances    = [best_score - s for s in scores]
        median_d     = np.median(distances)
        q1           = np.percentile(distances, 25)
        q3           = np.percentile(distances, 75)
        iqr_d        = q3 - q1
        alpha        = 1.0

        for i, s in cluster_scores.items():
            d = best_score - s
            if (d <= median_d + alpha * iqr_d) and (s >= 0.6 * best_score):
                selected_clusters.add(i)

    final_labels = []
    for i in range(k):
        is_mine = False
        if i in selected_clusters and metaLabels[i] == exLabel:
            # Physics constraints — must satisfy ALL of these
            ndvi_ok = centroids[i][ndvi_idx] <= 0.3      # stricter than 0.5
            bsi_ok  = centroids[i][bsi_idx]  >= 0.0      # BSI must be positive
            b12_ok  = centroids[i][b12_idx]  >= 0.0      # B12 must be above median
            if ndvi_ok and bsi_ok and b12_ok:
                is_mine = True
        final_labels.append(1 if is_mine else 0)

    best_k = k

    metadata = {
        "cluster_labels": {str(i): final_labels[i] for i in range(best_k)},
        "centroids": {
            str(i): {
                features[j]: float(centroids[i][j])
                for j in range(len(features))
            }
            for i in range(best_k)
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
def monitoringStart(startTime, endTime, mineid, windowSize, debug=0):
    mainDir = f"./Mine Data/Mine_{mineid}_data/"

    startTime       = ee.Date(startTime)
    actualStartTime = startTime.advance(-windowSize, "day")

    mine = geemap.geopandas_to_ee(gdf[gdf["mine_id"] == mineid])

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(actualStartTime, endTime)
        .filterBounds(mine)
        .map(mask)
    )

    s2 = featureEngineering(s2, windowSize).filterDate(startTime, endTime)

    # ----------------------------------------------------------------
    # MANUAL SCALING IN EARTH ENGINE
    # ----------------------------------------------------------------
    scaler = joblib.load(mainDir + "scaler.pkl")
    means  = scaler.center_.tolist()
    stds   = scaler.scale_.copy()
    stds[stds == 0] = 1.0
    stds = stds.tolist()

    eeScaling = scaling(means, stds)
    s2 = s2.map(eeScaling)

    # ----------------------------------------------------------------
    # MANUAL KMEANS ASSIGNMENT IN EARTH ENGINE
    # ----------------------------------------------------------------
    kmeans = joblib.load(mainDir + "kmeans.pkl")
    centroids = kmeans.cluster_centers_.tolist()

    with open(mainDir + "clusterData.json") as f:
        clusterData = json.load(f)

    clusterLabels = clusterData["cluster_labels"]
    exMaskCreate  = kmeansEE(centroids, clusterLabels)
    s2 = s2.map(exMaskCreate)

    s2 = s2.map(postprocessing)
    s2 = s2.sort("system:time_start")
    s2 = s2.map(lambda img: img.unmask(2))
    s2 = s2.toBands().clip(mine)

    task = ee.batch.Export.image.toDrive(
        image=s2,
        description=f"mine_{mineid}_excavation_{debug}",
        fileNamePrefix=f"mine_{mineid}_excavation_{debug}",
        region=mine.geometry(),
        scale=10,
        maxPixels=1e13
    )
    task.start()

# -------------------------------------------------------------------
# MONITORING PHASE (Part 2 - Offline Temporal Reasoning)
#
# This function:
# 1. Loads exported GeoTIFF(s)
# 2. Merges overlapping tiles per day
# 3. SAR gap-fill: fills cloud-masked periods using Sentinel-1
# 4. Builds a temporal confidence system
# 5. Produces final excavation timelines and visualizations
# 6. Generates no-go zone alerts if applicable
#
# SAR gap-fill is injected after retrieveDates() and before
# confidenceSystem() — the single point where cloud gaps exist
# as contiguous blocks of 2s in the Efixed array.
#
# sar_debug: pass the debug suffix used in sarMonitoringStart().
#            Set to None to skip SAR gap-fill entirely.
# sar_consecutive: number of SAR passes that must agree before
#            filling a gap. Default=2. Increase to 3 if candidate
#            area inflates unrealistically during monsoon months.
# -------------------------------------------------------------------
def monitoringComplete(mineid, threshold, debug=0, sar_debug=None, sar_consecutive=2):
    mainDir = f"./Mine Data/Mine_{mineid}_data/"
    outDir  = f"./Mine Data/Mine_{mineid}_data/Outputs/"

    alert_log_path = os.path.join(outDir, f"mine_{mineid}_alerts.log")

    if isinstance(debug, (int, str)):
        debug_list = [debug]
    else:
        debug_list = list(debug)

    all_E    = []
    all_dates = []
    transform = None
    crs       = None

    # ----------------------------------------------------------------
    # LOAD AND MERGE S2 RASTERS
    # ----------------------------------------------------------------
    for d in debug_list:
        tif_path = mainDir + f"mine_{mineid}_excavation_{d}.tif"

        with rasterio.open(tif_path) as src:
            E         = src.read()
            bandNames = src.descriptions

            if transform is None:
                transform = src.transform
                crs       = src.crs

        Efixed, dates = retrieveDates(bandNames, E)
        all_E.append(Efixed)
        all_dates.extend(dates)

    nogo = loadNogo(mineid, crs)

    if nogo is not None:
        with open(alert_log_path, "w") as f:
            f.write(f"# Monitoring run started: {datetime.now()}\n")

    Efixed = np.concatenate(all_E, axis=0)

    sort_idx = np.argsort(all_dates)
    Efixed   = Efixed[sort_idx]
    dates    = [all_dates[i] for i in sort_idx]

    # ----------------------------------------------------------------
    # SAR GAP-FILL
    #
    # Only activates where S2 has no valid observation (value == 2).
    # SAR GeoTIFF must be exported first via sarMonitoringStart().
    # Pass sar_debug=None to skip this step entirely.
    if sar_debug is not None:

        if isinstance(sar_debug, (int, str)):
            sar_debug_list = [sar_debug]
        else:
            sar_debug_list = list(sar_debug)

        for sd in sar_debug_list:

            sar_E, sar_dates = loadSAR(mineid, debug=sd)

            if sar_E is None:
                continue

            Efixed_before = Efixed.copy()

            Efixed = sarGapFill(
                Efixed,
                dates,
                sar_E,
                sar_dates,
                consecutive_required=sar_consecutive
            )

            SARCoveragePlot(
                mineid,
                outDir,
                dates,
                Efixed_before,
                Efixed
            )

            """" sarAnalysisStart(
                startTime=str(dates[0]),
                endTime=str(dates[-1]),
                mineid=mineid,
                s2_debug=debug_list[0],
                debug=sd
            )

            sarAnalysisPlot(
                mineid=mineid,
                debug=sd
            ) """

    # ----------------------------------------------------------------
    # CONFIDENCE SYSTEM
    # ----------------------------------------------------------------
    confirmed, candidate, confidence, firstSeen = confidenceSystem(
        Efixed, dates, threshold
    )

    retroConfirmed = retroConfirm(confirmed, firstSeen)

    # ----------------------------------------------------------------
    # OUTPUT GENERATION
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
        int(round(p / 100 * (len(dates) - 1)))
        for p in [0, 20, 40, 60, 80, 100]
    ]:
        debugCluster(mineid, i, dates, Efixed)

