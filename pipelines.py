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

from outputs import *

features = ["NDVI", "B11", "B12", "NDVI_median", "B11_median", "B12_median", "BSI_median", "NDMI_median", "NDVI_var", "NDMI_var", "NBR_slope"]

gdf = gpd.read_file("./Mine Polygons/SHP/mines_cils.shp").to_crs("EPSG:4326").reset_index(drop=True)
gdf["mine_id"] = gdf.index
gdf = gdf[["mine_id", "area", "perimeter", "geometry"]]

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

#Removes cloud shadows, cloud medium probability, cloud high probability, thin cirrus and snow/ice via masking
def mask(image):
    scl = image.select("SCL")
    invalid = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return image.updateMask(invalid.Not())

def addFeatures(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR")
    bsi = image.expression("((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))", {'SWIR' : image.select("B11"), 'RED' : image.select("B4"), 'NIR' : image.select("B8"), 'BLUE' : image.select("B2")}).rename("BSI")
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")

    return image.addBands([ndvi, nbr, bsi, ndmi]).select(["NDVI", "B11", "B12", "NBR", "BSI", "NDMI"])

def addTimeBand(image, t0):
    days = image.date().difference(t0, "day")
    time_band = ee.Image.constant(days).toFloat()
    return image.addBands(time_band.rename("time"))

def addSlope(s2, windowSize, band):
    def addSlopeActual(image):
        date = ee.Date(image.get("system:time_start"))
        window = s2.filterDate(date.advance(-windowSize, "day"), date).select(["time", band])
        count = window.size()

        slope = ee.Image(ee.Algorithms.If(count.gte(2), window.reduce(ee.Reducer.linearFit()).select("scale"), ee.Image.constant(0).toFloat().mask(ee.Image.constant(0).toFloat()))).rename(f"{band}_slope")
        slope = slope.clamp(-0.1, 0.1)

        return image.addBands(slope)
    return addSlopeActual

def addRollingStats(s2, windowSize):
    def addRollingStatsActual(image):
        date = ee.Date(image.get("system:time_start"))
        window = s2.filterDate(date.advance(-windowSize, "day"), date)
        count = window.size()

        dummy_var = ee.Image.constant(0).toFloat()
        dummy_median = ee.Image.constant([0, 0, 0, 0, 0]).toFloat()

        ndviVar = ee.Algorithms.If(count.gte(2),window.select('NDVI').reduce(ee.Reducer.variance()), dummy_var)
        ndviVar = ee.Image(ndviVar).rename("NDVI_var") 

        ndmiVar = ee.Algorithms.If(count.gte(2), window.select('NDMI').reduce(ee.Reducer.variance()),dummy_var)
        ndmiVar = ee.Image(ndmiVar).rename("NDMI_var")

        median = ee.Algorithms.If(count.gte(1), window.select(['NDVI', 'B11', 'B12', 'BSI', "NDMI"]).median(), dummy_median )
        median = ee.Image(median).rename(["NDVI_median", "B11_median", "B12_median", "BSI_median", "NDMI_median"])

        return image.addBands([ndviVar, ndmiVar, median])

    return addRollingStatsActual

def featureEngineering(s2, windowSize):
    s2 = s2.map(addFeatures)
    t0 = ee.Date(s2.first().get("system:time_start"))
    s2 = s2.map(lambda img: addTimeBand(img, t0))
    addRollingStatsUse = addRollingStats(s2, windowSize)
    s2 = s2.map(addRollingStatsUse)
    addSlopeUse = addSlope(s2, windowSize, "NBR")
    s2 = s2.map(addSlopeUse)
    return s2.select(features)

def imageToTable(mine):
    def imageToTableActual(image):
        return image.sample(region = mine, scale = 10, geometries = False)
    return imageToTableActual

#We create an "excavated-score" consisting of having low NDVI, high BSI and SWIR indices
def findExcavated(centroids, labels):
    b12_idx = features.index("B12_median")  
    bsi_idx = features.index("BSI_median")  
    ndvi_idx = features.index("NDVI_median") 
    ndmi_idx = features.index("NDMI_median")

    scores = []
    
    for group_id in [0, 1]:
        group_indices = [i for i, x in enumerate(labels) if x == group_id]
        
        if len(group_indices) == 0:
            scores.append(-np.inf)
            continue

        mean_b12 = np.mean([centroids[i][b12_idx] for i in group_indices])
        mean_bsi = np.mean([centroids[i][bsi_idx] for i in group_indices])
        mean_ndvi = np.mean([centroids[i][ndvi_idx] for i in group_indices])
        mean_ndmi = np.mean([centroids[i][ndmi_idx] for i in group_indices])

        score = (2.0 * mean_b12) + (1.0 * mean_bsi) - (0.5 * mean_ndvi) -(1.0 * mean_ndmi)
        scores.append(score)

    exLabel = 0 if scores[0] > scores[1] else 1
    return exLabel, scores

def jsonMaker(centroids, labels, exLabel, means, dir):
    clusterLabels = {}
    for i in range(len(centroids)):
        clusterLabels[str(i)] = (1 if labels[i] == exLabel else 0)

    centroidData = {}
    for i, row in enumerate(centroids):
        centroidData[str(i)] = {features[j]: float(row[j]) for j in range(len(features))}

    metadata = {
        "cluster_labels" : clusterLabels, 
        "centroids" : centroidData, 
        "metacluster_stats": {
            "0": { "excavation_score": float(means[0]) },
            "1": { "excavation_score": float(means[1]) }
        },
        "excavated_metacluster" : exLabel
    }

    with open(dir + "clusterData.json", "w") as f:
        json.dump(metadata, f, indent = 2)

def scaling(means, stds):
    def scalingActual(image):
           meanImg = ee.Image.constant(means)
           stdImg = ee.Image.constant(stds)

           #Applying StandardScaling
           return image.subtract(meanImg).divide(stdImg)
    return scalingActual

#calculate distance to a centroid, this is what we have to minimize to find the closest centroid
def distCentroid(img, centroid):
    return img.subtract(centroid).pow(2).reduce(ee.Reducer.sum())

def kmeansEE(centroids, clusterLabels):
    def kmeansEEActual(image):
        distImgs = [distCentroid(image, ee.Image.constant(c)) for c in centroids]

        stack = ee.Image.cat(distImgs)
        #Argmin doesnt exist so we negate the array to use argmax instead lmaoo
        negDistArray = stack.toArray().multiply(-1)
        clusterId = negDistArray.arrayArgmax().arrayGet([0])
        
        clusterIds = [int(k) for k in clusterLabels.keys()]
        excavatedStat = [clusterLabels[k] for k in clusterLabels.keys()]

        excavated = clusterId.remap(clusterIds, excavatedStat)
        mask1 = image.mask().select(0)
        return excavated.updateMask(mask1).copyProperties(image, ["system:time_start"])
    return kmeansEEActual

#Once again, we dont have access to skimage inside of EE, so we gotta settle for this 
def postprocessing(image):
    opened = image.focalMin(radius = 1, units = "pixels").focalMax(radius = 1, units = "pixels")
    closed = opened.focalMax(radius = 1, units = "pixels").focalMin(radius = 1, units = "pixels")
    return closed

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
        while j < len(dates) and dates[j] == current_date:
            day_imgs.append(E[j])
            j += 1

        stack = np.stack(day_imgs)  # shape (N, H, W)

        has_excavation = np.any(stack == 1, axis=0)
        has_known = np.any(stack != 2, axis=0)

        merged = np.full(stack.shape[1:], 2, dtype=np.uint8)
        merged[has_excavation] = 1
        merged[(~has_excavation) & has_known] = 0

        merged_E.append(merged)
        merged_dates.append(current_date)

        i = j

    merged_E = np.stack(merged_E)

    return merged_E, merged_dates

def confidenceSystem(E, dates, threshold):
    T, H, W = E.shape
    total_pixels = H * W

    confirmed = np.zeros((T, H, W), dtype = np.uint8)
    candidate = np.zeros((T, H, W), dtype = np.uint8)
    confidence = np.zeros((T, H, W), dtype = np.float32)

    daysExcavated = np.zeros((H, W), dtype = np.int32)
    firstSeen = np.full((H, W), -1, dtype = np.int32)
    confirmedFirstSeen = np.full((H, W), -1, dtype=np.int32)

    spike_limit = 0.20 * total_pixels

    for t in range(T):
        Et = E[t].copy()

        if t > 0:
            confirmed_area = np.count_nonzero(confirmed[t-1])
            current_raw_area = np.count_nonzero(Et == 1)
    
            if confirmed_area > 0:
                growth = current_raw_area - confirmed_area
                if growth > spike_limit:
                    new_growth_mask = (Et == 1) & (confirmed[t-1] == 0)
                    Et[new_growth_mask] = 0

        deltaDays = 0 if t == 0 else (dates[t] - dates[t-1]).days
        prevConfirmed = np.zeros((H, W), dtype = np.uint8) if t == 0 else confirmed[t-1]
    
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

        newly_confirmed_indices = np.where(newConfirmed)
        current_starts = firstSeen[newly_confirmed_indices]
        confirmedFirstSeen[newly_confirmed_indices] = current_starts

        candidate[t] = ((Et == 1) | (confirmed[t] == 1)).astype(np.uint8)

        confidence[t] = np.clip(daysExcavated/threshold, 0, 1)
        confidence[t][confirmed[t] == 1] = 1

    return confirmed, candidate, confidence, confirmedFirstSeen

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

#During training, we could skip the first windowsize images, because we need time for the rolling stats to develop
#Instead also what we could do is calculate stats from start-windowsize, so by the time we reach the start, the stats would have developed
def trainingStart(startTime, endTime, mineid, windowSize, debug = 0):
    startTime = ee.Date(startTime)
    actualStartTime = startTime.advance(-windowSize, "day")
    mine = geemap.geopandas_to_ee(gdf[gdf["mine_id"] == mineid])

    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(actualStartTime, endTime).filterBounds(mine).map(mask)
    s2 = featureEngineering(s2, windowSize).filterDate(startTime, endTime)

    imageToTableUse = imageToTable(mine)
    s2 = s2.map(imageToTableUse).flatten().select(features)

    task = ee.batch.Export.table.toDrive(collection = s2, description = f"mine_{mineid}_features_{debug}", fileFormat = "CSV")
    task.start()

def trainingComplete(mineid, debug = 0, k = 6):
    mainDir = f"./Mine Data/Mine_{mineid}_data/"
    df = pd.read_csv(mainDir + f"mine_{mineid}_features_{debug}.csv")
    df = df[features]

    scaler = RobustScaler(quantile_range=(25, 75))
    scaler.fit(df)
    #NBR Slope be going brr, so we apply a hard threshold so the values dont skyrocket
    slope_idx = features.index("NBR_slope")
    scaler.scale_[slope_idx] = 0.05
    var_idx = features.index("NDVI_var")
    if scaler.scale_[var_idx] < 0.5: 
        scaler.scale_[var_idx] = 1.0
    ndmi_var_idx = features.index("NDMI_var")
    if scaler.scale_[ndmi_var_idx] < 0.5:
        scaler.scale_[ndmi_var_idx] = 1.0
    dfScaled = scaler.transform(df)
    joblib.dump(scaler, mainDir + "scaler.pkl")
   
    kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 10)
    kmeans.fit(dfScaled)
    joblib.dump(kmeans, mainDir + "kmeans.pkl")

    #Water and shadows have really low SWIR indices which is confusing our model, since really low SWIR and low SWIR are both "not mine", we can apply a filter to make sure water doesnt interfere with our metaclustering
    centroids = kmeans.cluster_centers_
    centroids_for_meta = centroids.copy()
    swir_indices = [ features.index("B11"),  features.index("B12"),  features.index("B11_median"), features.index("B12_median"), features.index("NDMI_median")]
    for idx in swir_indices:
        centroids_for_meta[:, idx] = np.maximum(centroids_for_meta[:, idx], -1.0)

    metaKMeans = KMeans(n_clusters = 2, random_state = 42, n_init = 10)
    metaLabels = metaKMeans.fit_predict(centroids_for_meta)

    exLabel, means = findExcavated(centroids, metaLabels)  

    b12_idx = features.index("B12")
    ndvi_idx = features.index("NDVI")
    b12_med_idx = features.index("B12_median")
    ndvi_med_idx = features.index("NDVI_median")
    
    final_labels = metaLabels.copy()

    for i in range(k):
        if metaLabels[i] == exLabel:
            
            current_score = centroids[i][b12_idx] - centroids[i][ndvi_idx]
            
            median_score = centroids[i][b12_med_idx] - centroids[i][ndvi_med_idx]

            was_nature = median_score < -0.2 
            is_weak = current_score < 0.6
            
            if was_nature or is_weak:
                print(f"PRUNING CLUSTER {i}: Identified as Misclassification (Hist: {median_score:.2f}, Curr: {current_score:.2f}) -> FLIPPED TO SAFE")
                final_labels[i] = 1 - exLabel 

    jsonMaker(centroids, final_labels, exLabel, means, mainDir)
    
def monitoringStart(startTime, endTime, mineid, windowSize, debug = 0):
    mainDir = f"./Mine Data/Mine_{mineid}_data/"
    startTime = ee.Date(startTime)
    actualStartTime = startTime.advance(-windowSize, "day")
    mine = geemap.geopandas_to_ee(gdf[gdf["mine_id"] == mineid])

    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(actualStartTime, endTime).filterBounds(mine).map(mask)
    s2 = featureEngineering(s2, windowSize).filterDate(startTime, endTime)

#We would like to export into a nice table that we can reclassify like last time, but we would also want to utilize EE for speed and ease
#We will need the power of EE to do computations to create our excavated binary mask at each time t and stuff
#So we need to settle for manually applying scaling and clustering through EE on our pixels BEFORE exporting
    scaler = joblib.load(mainDir + "scaler.pkl")
    means = scaler.center_.tolist()
    stds = scaler.scale_.copy()
    stds[stds == 0] = 1.0
    stds = stds.tolist()
    eeScaling = scaling(means, stds)
    s2 = s2.map(eeScaling)

    kmeans = joblib.load(mainDir + "kmeans.pkl")
    centroids = kmeans.cluster_centers_.tolist()
    with open(mainDir + "clusterData.json") as f:
        clusterData = json.load(f)
    clusterLabels = clusterData["cluster_labels"]
    exMaskCreate = kmeansEE(centroids, clusterLabels)
    s2 = s2.map(exMaskCreate)

    s2 = s2.map(postprocessing)

    s2 = s2.sort("system:time_start")
    s2 = s2.map(lambda img: img.unmask(2))

    s2 = s2.toBands().clip(mine)

    task = ee.batch.Export.image.toDrive(image = s2, description = f"mine_{mineid}_excavation_{debug}", fileNamePrefix = f"mine_{mineid}_excavation_{debug}", region = mine.geometry(), scale = 10, maxPixels = 1e13)
    task.start()

def monitoringComplete(mineid, threshold, debug = 0, nogo = None):
    mainDir = f"./Mine Data/Mine_{mineid}_data/"
    outDir = f"./Mine Data/Mine_{mineid}_data/Outputs/"

    if isinstance(debug, (int, str)):
        debug_list = [debug]
    else:
        debug_list = list(debug)

    all_E = []
    all_dates = []
    transform = None
    crs = None

    for d in debug_list:
        tif_path = mainDir + f"mine_{mineid}_excavation_{d}.tif"

        with rasterio.open(tif_path) as src:
            E = src.read()
            bandNames = src.descriptions

            if transform is None:
                transform = src.transform
                crs = src.crs

        Efixed, dates = retrieveDates(bandNames, E)

        all_E.append(Efixed)
        all_dates.extend(dates)

    Efixed = np.concatenate(all_E, axis=0)

    sort_idx = np.argsort(all_dates)
    Efixed = Efixed[sort_idx]
    dates = [all_dates[i] for i in sort_idx]
   
    confirmed, candidate, confidence, firstSeen = confidenceSystem(Efixed, dates, threshold)
    retroConfirmed = retroConfirm(confirmed, firstSeen)

    makeSpatialMaps(mineid, outDir, dates, retroConfirmed, confidence, transform, crs, nogo)
    retroConfirmedAnalysis(mineid, dates, outDir, retroConfirmed, transform, crs, nogo)
    retroAreas = ExcavationTimePlot(mineid, outDir, retroConfirmed, dates, transform) 
    candAreas = CandidateExcavationTimePlot(mineid, outDir, candidate, dates, transform)
    growthRate = GrowthRatePlot(mineid, outDir, retroConfirmed, dates, transform)
    ComparisionPlot(mineid, outDir, retroAreas, candAreas)
    NormalizedExcavationPlot(mineid, outDir, retroAreas)
    FirstSeenDateMap(mineid, outDir, firstSeen, dates, transform, crs, nogo)

    NoGoAlertSystem(mineid, outDir, dates, candidate, confirmed, transform, crs, nogo)
    NoGoExcavationTimePlot(mineid, outDir, dates, candidate, confirmed, transform, crs, nogo)

    for i in [int(round(p/100 * (len(dates)-1))) for p in [0, 20, 40, 60, 80, 100]]:
        debugCluster(mineid, i, dates, Efixed)

def scanForActivity(mine_ids, reference_mine_id):
    mainDir = f"./Mine Data/Mine_{reference_mine_id}_data/"

    scaler = joblib.load(mainDir + "scaler.pkl")
    means = scaler.center_.tolist()
    stds = scaler.scale_.copy()
    stds[stds == 0] = 1.0
    stds = stds.tolist()

    kmeans = joblib.load(mainDir + "kmeans.pkl")
    centroids = kmeans.cluster_centers_.tolist()

    with open(mainDir + "clusterData.json") as f:
        clusterData = json.load(f)
    clusterLabels = clusterData["cluster_labels"]

    target_mines = gdf[gdf["mine_id"].isin(mine_ids)]
    ee_mines = geemap.geopandas_to_ee(target_mines)

    combined_bounds = ee_mines.geometry()

    def get_classified_area_image(year_start, year_end, output_name):
        start_date = ee.Date(year_start)
        end_date = ee.Date(year_end)
        buffer_start = start_date.advance(-65, "day")

        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterDate(buffer_start, end_date)
              .filterBounds(combined_bounds)
              .map(mask))

        s2_features = featureEngineering(s2, windowSize=60)
        
        composite = (s2_features
                     .filterDate(start_date, end_date)
                     .median()
                     .unmask(0)
                     .toFloat())
        
        scaled = scaling(means, stds)(composite)
        classified = ee.Image(kmeansEE(centroids, clusterLabels)(scaled))
        cleaned = postprocessing(classified)

        return cleaned.eq(1).multiply(ee.Image.pixelArea()).rename(output_name)

    img_2022 = get_classified_area_image("2022-01-01", "2022-12-31", "area_2022_m2")
    
    img_2024 = get_classified_area_image("2024-01-01", "2024-12-31", "area_2024_m2")

    combined_img = img_2022.addBands(img_2024)

    stats = combined_img.reduceRegions(
        collection=ee_mines,
        reducer=ee.Reducer.sum(),
        scale=30,
        tileScale=16  
    )

    def compute_growth(feature):
        a22 = ee.Number(feature.get("area_2022_m2"))
        a24 = ee.Number(feature.get("area_2024_m2"))
        
        growth_m2 = a24.subtract(a22)
        
        growth_pct = ee.Algorithms.If(
            a22.gt(0),
            growth_m2.divide(a22).multiply(100),
            0
        )
        
        status = ee.Algorithms.If(growth_m2.gt(5000), "ACTIVE", "INACTIVE")
        
        return feature.set({
            "growth_m2": growth_m2,
            "growth_pct": growth_pct,
            "status": status
        })

    final_results = stats.map(compute_growth)

    task = ee.batch.Export.table.toDrive(
        collection=final_results,
        description=f"MineActivityScan_Ref_{reference_mine_id}",
        fileFormat="CSV",
        selectors=[
            "mine_id", "status", "growth_m2", "growth_pct", 
            "area_2022_m2", "area_2024_m2"
        ]
    )

    task.start()