import ee
import geopandas as gpd
import pandas as pd
import geemap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import json

features = ["NDVI", "B11", "B12", "NDVI_median", "B11_median", "B12_median", "BSI_median", "NDVI_var", "NBR_slope"]

gdf = gpd.read_file("./Mine Polygons/SHP/mines_cils.shp").to_crs("EPSG:4326").reset_index(drop=True)
gdf["mine_id"] = gdf.index
gdf = gdf[["mine_id", "area", "perimeter", "geometry"]]

#Removes cloud shadows, cloud medium probability, cloud high probability, thin cirrus and snow/ice via masking
def mask(image):
    scl = image.select("SCL")
    invalid = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return image.updateMask(invalid.Not())

def addFeatures(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR")
    bsi = image.expression("((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))", {'SWIR' : image.select("B11"), 'RED' : image.select("B4"), 'NIR' : image.select("B8"), 'BLUE' : image.select("B2")}).rename("BSI")

    return image.addBands([ndvi, nbr, bsi]).select(["NDVI", "B11", "B12", "NBR", "BSI"])

def addTimeBand(image):
    time = ee.Image.constant(0).add(image.date().millis()).rename("time").toInt64()
    return image.addBands(time)

def addSlope(s2, windowSize, band):
    def addSlopeActual(image):
        date = ee.Date(image.get("system:time_start"))
        window = s2.filterDate(date.advance(-windowSize, "day"), date).select(["time", band])

        fit = window.reduce(ee.Reducer.linearFit())
        slope = fit.select("scale").rename(f"{band}_slope")
        return image.addBands(slope)
    return addSlopeActual

def addRollingStats(s2, windowSize):
    def addRollingStatsActual(image):
        date = ee.Date(image.get("system:time_start"))
        window = s2.filterDate(date.advance(-windowSize, "day"), date)

        medianOnes = window.select(['NDVI', 'B11', 'B12', 'BSI'])
        ndviVar = window.select('NDVI')

        ndvi_var =  ndviVar.reduce(ee.Reducer.variance()).rename(["NDVI_var"])
        median = medianOnes.median().rename(["NDVI_median", "B11_median", "B12_median", "BSI_median"])

        return image.addBands([ndvi_var, median])

    return addRollingStatsActual

def featureEngineering(s2, windowSize):
    s2 = s2.map(addFeatures)
    s2 = s2.map(addTimeBand)
    addRollingStatsUse = addRollingStats(s2, windowSize)
    s2 = s2.map(addRollingStatsUse)
    addSlopeUse = addSlope(s2, windowSize, "NBR")
    s2 = s2.map(addSlopeUse)
    return s2.select(features)

def imageToTable(mine):
    def imageToTableActual(image):
        return image.sample(region = mine, scale = 10, geometries = False)
    return imageToTableActual

#Safe to assume that the clusters with lower mean NDVI are the excavated pixel clusters
def findExcavated(centroids, labels):
    ndvi = centroids[:, features.index("NDVI")]

    mean0 = ndvi[labels == 0].mean()
    mean1 = ndvi[labels == 1].mean()

    exLabel = 0 if mean0 < mean1 else 1
    return exLabel, [mean0, mean1]

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
            "0": { "mean_NDVI": float(means[0]) },
            "1": { "mean_NDVI": float(means[1]) }
        },
        "excavated_metacluster" : exLabel
    }

    with open(dir + "clusterData.json", "w") as f:
        json.dump(metadata, f, indent = 2)

#During training, we could skip the first windowsize images, because we need time for the rolling stats to develop
#Instead also what we could do is calculate stats from start-windowsize, so by the time we reach the start, the stats would have developed
def trainingStart(startTime, endTime, mineid, windowSize):
    startTime = ee.Date(startTime)
    actualStartTime = startTime.advance(-windowSize, "day")
    mine = geemap.geopandas_to_ee(gdf[gdf["mine_id"] == mineid])

    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(actualStartTime, endTime).filterBounds(mine).map(mask)
    s2 = featureEngineering(s2, windowSize).filterDate(startTime, endTime)

    imageToTableUse = imageToTable(mine)
    s2 = s2.map(imageToTableUse).flatten().select(features)

    task = ee.batch.Export.table.toDrive(collection = s2, description = f"mine_{mineid}_features", fileFormat = "CSV")
    task.start()

def trainingComplete(mineid, k = 6):
    mainDir = f"./Mine Data/Mine_{mineid}_data/"
    df = pd.read_csv(mainDir + f"mine_{mineid}_features.csv")
    df = df[features]

    scaler = StandardScaler()
    dfScaled = scaler.fit_transform(df)
    joblib.dump(scaler, mainDir + "scaler.pkl")
   
    kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 10)
    kmeans.fit(dfScaled)
    joblib.dump(kmeans, mainDir + "kmeans.pkl")

    centroids = kmeans.cluster_centers_
    metaKMeans = KMeans(n_clusters = 2, random_state = 42, n_init = 10)
    metaLabels = metaKMeans.fit_predict(centroids)

    exLabel, means = findExcavated(centroids, metaLabels)  
    jsonMaker(centroids, metaLabels, exLabel, means, mainDir)

def monitoring():
    pass

