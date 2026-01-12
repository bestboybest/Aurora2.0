import ee
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from rasterio.features import shapes, geometry_mask
from rasterio.transform import array_bounds
from shapely.geometry import shape
import geemap
from PIL import Image
import urllib.request
import io

gdf = gpd.read_file("./Mine Polygons/SHP/mines_cils.shp").to_crs("EPSG:4326").reset_index(drop=True)
gdf["mine_id"] = gdf.index
gdf = gdf[["mine_id", "area", "perimeter", "geometry"]]

titleFont = {'weight':'bold', 'color':'orangered', 'size':20, 'name':'Comic Sans MS'}
normalFont = {'color':'maroon', 'size':16}

def polygonize(mask, transform, crs):
    geometries = []
    values = []

    for geom, value in shapes(mask.astype(np.uint8), transform = transform):
        if value == 1:
            geometries.append(shape(geom))
            values.append(value)
    
    exgdf = gpd.GeoDataFrame({"value" : values}, geometry = geometries, crs = crs)
    return exgdf

def eeToNumpy(image, region, vis, size=1024):
    url = image.getThumbURL({
        "region": region,
        "dimensions": size,
        "format": "png",
        **vis
    })

    with urllib.request.urlopen(url) as response:
        img = Image.open(io.BytesIO(response.read()))
        return np.array(img)
    
def rasterize_nogo(nogo_gdf, shape, transform):

    return geometry_mask(
        nogo_gdf.geometry,
        out_shape=shape,
        transform=transform,
        invert=True
    )
    
def debugCluster(mineid, t, dates, E):
    eeDate = ee.Date(dates[t].isoformat())
    mine = geemap.geopandas_to_ee(gdf[gdf["mine_id"] == mineid])
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(mine).filterDate(eeDate.advance(-10, "day"), eeDate.advance(10, "day")).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)).median().clip(mine))

    rgb = eeToNumpy( s2,region=mine.geometry(), vis={"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000} )

    # --- Raw excavation mask ---
    raw_exc = E[t]
    raw_exc = np.ma.masked_where(raw_exc != 1, raw_exc)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    minx, miny, maxx, maxy = gdf[gdf["mine_id"] == mineid].total_bounds
    extent = [minx, maxx, miny, maxy]

    ax1.imshow(rgb, extent=extent, origin="upper")
    gdf[gdf["mine_id"] == mineid].boundary.plot(ax=ax1, edgecolor="black", linewidth=2)
    ax1.set_title(f"RGB Sentinel-2 | {dates[t].isoformat()}", fontdict=titleFont)
    ax1.axis("off")

    ax2.imshow(raw_exc, cmap="Reds", extent=extent, origin="upper")
    gdf[gdf["mine_id"] == mineid].boundary.plot(ax=ax2, edgecolor="black", linewidth=2)
    ax2.set_title(f"Raw Cluster Excavation (t={t})", fontdict=titleFont)
    ax2.axis("off")

    plt.tight_layout()
    plt.show()

def makeSpatialMaps(mineid, dir, dates, retroConfirmed, confidence, transform, crs, nogo = None):
    T = len(dates)

    percentiles = [0, 25, 50, 75, 100]
    t_idxs = [int(round(p / 100 * (T - 1))) for p in percentiles]

    retroPolys = [
        polygonize(retroConfirmed[t], transform, crs)
        for t in t_idxs
    ]

    for i, (t, p) in enumerate(zip(t_idxs, percentiles)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        mine = geemap.geopandas_to_ee(gdf[gdf["mine_id"] == mineid])
        eeDate = ee.Date(dates[t].isoformat())
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(mine).filterDate(eeDate.advance(-10, "day"), eeDate.advance(10, "day")).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)).median().clip(mine))

        rgb = eeToNumpy(s2,region=mine.geometry(),vis={"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000})

        minx, miny, maxx, maxy = gdf[gdf["mine_id"] == mineid].total_bounds
        extent = [minx, maxx, miny, maxy]
        ax1.set_xlim(minx, maxx)
        ax1.set_ylim(miny, maxy)
        ax1.imshow(rgb, extent=extent, origin="upper", interpolation="nearest", vmin=0, vmax=255, alpha=0.85)
        ax1.set_aspect("equal")

        poly = retroPolys[i].to_crs("EPSG:4326")
        if not poly.empty and poly.is_valid.any():
            poly.boundary.plot(ax=ax1, edgecolor="cyan", linewidth=2, zorder=10)

        gdf[gdf["mine_id"] == mineid].boundary.plot(ax=ax1, edgecolor="black", linewidth=2)
        if nogo is not None:
            nogo.boundary.plot(ax=ax1, edgecolor="yellow", linewidth=2, linestyle="--")

        ax1.set_title(f"Retroconfirmed Excavated Area – {dates[t].isoformat()}", fontdict=titleFont)
        ax1.axis("off")

        conf = confidence[t]
        conf_masked = np.ma.masked_where(conf == 0, conf)
        minegdf = gdf[gdf["mine_id"] == mineid].to_crs(crs)
        H, W = conf_masked.shape
        bounds = array_bounds(H, W, transform)
        bounds = [bounds[0], bounds[2], bounds[1], bounds[3]]

        mine_mask = geometry_mask(minegdf.geometry,out_shape=conf_masked.shape, transform=transform, invert=True)
        conf_vis = np.ma.masked_where(~mine_mask, conf_masked)

        im = ax2.imshow(conf_vis, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest", aspect="equal", extent=bounds,origin="upper" )

        ax2.set_title(f"Confidence on Excavation – {dates[t].isoformat()}", fontdict=titleFont)
        minegdf.boundary.plot(ax=ax2, edgecolor="black", linewidth=2)
        ax2.axis("off")
        plt.colorbar(im, ax=ax2, fraction=0.046)

        plt.savefig(
    dir + f"mine_{mineid}_spatialMap_{p}percent.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.2
)
        plt.close()


def retroConfirmedAnalysis(mineid, dates, dir, retroConfirmed, transform, crs, nogo = None):
    T = len(dates)

    percentiles = list(range(0, 101, 10))  # 0,10,20,...,100
    t_idxs = [int(round(p/100 * (T-1))) for p in percentiles]

    minegdf = gdf[gdf["mine_id"] == mineid]
    mine = geemap.geopandas_to_ee(minegdf)

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    axes = axes.flatten()

    minx, miny, maxx, maxy = minegdf.total_bounds
    extent = [minx, maxx, miny, maxy]

    for ax, t, p in zip(axes, t_idxs, percentiles):
        date = dates[t]

        eeDate = ee.Date(date.isoformat())
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(mine).filterDate(eeDate.advance(-10, "day"), eeDate.advance(10, "day")).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)).median().clip(mine))

        rgb = eeToNumpy(s2,region=mine.geometry(),vis={"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000})

        ax.imshow(rgb, extent=extent, origin="upper", interpolation="nearest", alpha=0.85)

        poly = polygonize(retroConfirmed[t], transform, crs).to_crs("EPSG:4326")
        if not poly.empty and poly.is_valid.any():
            poly.boundary.plot(ax=ax, edgecolor="cyan", linewidth=2, zorder=10)

        minegdf.boundary.plot(ax=ax, edgecolor="black", linewidth=2)

        if nogo is not None:
            nogo.boundary.plot(ax=ax, edgecolor="yellow", linewidth=2, linestyle="--")

        ax.set_title(f"{p} percentile\n{date.isoformat()}", fontdict=titleFont)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ax.axis("off")

    for ax in axes[len(percentiles):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(dir + f"mine_{mineid}_excavationProgress.png", dpi=300)
    plt.close()

def ExcavationTimePlot(mineid, dir, retroConfirmed, dates, transform):
    pixel_area_m2 = abs(transform.a * transform.e)

    areas = []
    for t in range(len(dates)):
        excavated_pixels = np.count_nonzero(retroConfirmed[t] == 1)
        area = excavated_pixels * pixel_area_m2
        areas.append(area)

    df = pd.DataFrame({"date": pd.to_datetime(dates),"excavated_area_m2": areas})

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["date"],
        df["excavated_area_m2"],
        linewidth=2
    )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))

    ax.set_xlabel("Year", fontdict=normalFont)
    ax.set_ylabel("Excavated Area (m²)", fontdict=normalFont)
    ax.set_title(
        "Retroconfirmed Excavated Area vs Time (2018–2025)",
        fontdict=titleFont
    )

    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    plt.tight_layout()
    plt.savefig(dir + f"mine_{mineid}_AreavsTime.png", dpi=300)
    plt.close()

    return df

def CandidateExcavationTimePlot(mineid, dir, candidate, dates, transform):
    pixel_area_m2 = abs(transform.a * transform.e)

    areas = []
    for t in range(len(dates)):
        candidate_pixels = np.count_nonzero(candidate[t] == 1)
        area = candidate_pixels * pixel_area_m2
        areas.append(area)

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "candidate_excavated_area_m2": areas
    })

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["date"],
        df["candidate_excavated_area_m2"],
        linewidth=2,
        color="orange"
    )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))

    ax.set_xlabel("Year", fontdict=normalFont)
    ax.set_ylabel("Candidate Excavated Area (m²)", fontdict=normalFont)
    ax.set_title("Candidate Excavated Area vs Time (Unconfirmed Signals)",fontdict=titleFont)

    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    plt.tight_layout()
    plt.savefig(dir + f"mine_{mineid}_CandidateAreavsTime.png", dpi=300)
    plt.close()

    return df

def GrowthRatePlot(mineid, dir, retroConfirmed, dates, transform):
    pixel_area_m2 = abs(transform.a * transform.e)

    areas = []
    for t in range(len(dates)):
        excavated_pixels = np.count_nonzero(retroConfirmed[t] == 1)
        areas.append(excavated_pixels * pixel_area_m2)

    rates = [0.0]
    for i in range(1, len(areas)):
        delta_days = (dates[i] - dates[i-1]).days
        if delta_days == 0:
            rates.append(0.0)
        else:
            rates.append((areas[i] - areas[i-1]) / delta_days)

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "growth_rate_m2_per_day": rates
    })

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["date"], df["growth_rate_m2_per_day"], linewidth=2, color="purple")

    ax.axhline(0, color="black", linewidth=1, alpha=0.6)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))

    ax.set_xlabel("Year", fontdict=normalFont)
    ax.set_ylabel("Excavation Growth Rate (m²/day)", fontdict=normalFont)
    ax.set_title("Excavation Growth Rate vs Time", fontdict=titleFont)

    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    plt.tight_layout()
    plt.savefig(dir + f"mine_{mineid}_GrowthRate.png", dpi=300)
    plt.close()

    return df

def ComparisionPlot(mineid, dir, retro_df, candidate_df):

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(retro_df["date"], retro_df["excavated_area_m2"],linewidth=3,color="tab:blue",label="Retroconfirmed Excavation")

    ax.plot(candidate_df["date"],candidate_df["candidate_excavated_area_m2"],linewidth=2,linestyle="--",color="tab:orange", label="Candidate Excavation")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))

    ax.set_xlabel("Year", fontdict=normalFont)
    ax.set_ylabel("Excavated Area (m²)", fontdict=normalFont)
    ax.set_title("Candidate vs Retroconfirmed Excavated Area",fontdict=titleFont)

    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    ax.legend()
    plt.tight_layout()
    plt.savefig(dir + f"mine_{mineid}_ComparisionPlot.png", dpi=300)
    plt.close()

def NormalizedExcavationPlot(mineid, dir, retro_df):
    
    mine_area_m2 = (gdf[gdf["mine_id"] == mineid].to_crs("EPSG:6933").geometry.area.values[0])

    dfy = retro_df.copy()
    normalized = dfy["excavated_area_m2"] / mine_area_m2

    df = pd.DataFrame({"date": retro_df["date"],"normalized_excavation": normalized})

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["date"], df["normalized_excavation"], linewidth=3, color="tab:green")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))

    ax.set_xlabel("Year", fontdict=normalFont)
    ax.set_ylabel("Normalized Excavation (fraction of mine)", fontdict=normalFont)
    ax.set_title("Normalized Retroconfirmed Excavation vs Time", fontdict=titleFont)

    ax.set_ylim(0, 1)
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    plt.tight_layout()
    plt.savefig(dir + f"mine_{mineid}_NormalizedExcavation.png", dpi=300)
    plt.close()

    df.to_csv(dir + f"mine_{mineid}_ExcavationIntensity.csv", index=False)

    return df

def FirstSeenDateMap(mineid, dir, confirmedFirstSeen, dates, transform, crs, nogo=None):
    H, W = confirmedFirstSeen.shape

    start_date = dates[0]
    first_detection_days = np.full((H, W), np.nan, dtype=np.float32)

    for y in range(H):
        for x in range(W):
            t0 = confirmedFirstSeen[y, x]
            if t0 >= 0:
                first_detection_days[y, x] = (dates[t0] - start_date).days

    detection_masked = np.ma.masked_invalid(first_detection_days)

    fig, ax = plt.subplots(figsize=(9, 9))

    bounds = array_bounds(H, W, transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    im = ax.imshow(
        detection_masked,
        cmap="viridis",
        interpolation="nearest",
        extent=extent,
        origin="upper"
    )

    minegdf = gdf[gdf["mine_id"] == mineid].to_crs(crs)
    minegdf.boundary.plot(ax=ax, edgecolor="black", linewidth=2)

    if nogo is not None:
        nogo.to_crs(crs).boundary.plot(ax=ax, edgecolor="red", linewidth=2, linestyle="--")

    ax.set_title(
        "First Confirmed Excavation Date (Days Since Start)",
        fontdict=titleFont,
        pad=12  
    )
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Days Since Monitoring Start", fontdict=normalFont)

    plt.savefig(
        dir + f"mine_{mineid}_FirstSeenPlot.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2
    )
    plt.close()

    return first_detection_days

def NoGoAlertSystem(mineid, outDir,  dates, candidate,  confirmed, transform, crs, nogo=None,expansion_threshold_pixels=20):

    if nogo is None:
        return  

    alert_path = outDir + f"mine_{mineid}_alerts.log"

    H, W = candidate.shape[1:]
    pixel_area_m2 = abs(transform.a * transform.e)

    nogo = nogo.to_crs(crs)
    nogo_mask = rasterize_nogo(nogo, (H, W), transform)

    prev_confirmed_area = 0

    with open(alert_path, "a") as log:
        for t in range(len(dates)):
            date = dates[t]

            cand_mask = (candidate[t] == 1) & nogo_mask
            conf_mask = (confirmed[t] == 1) & nogo_mask

            cand_pixels = np.count_nonzero(cand_mask)
            conf_pixels = np.count_nonzero(conf_mask)

            cand_area = cand_pixels * pixel_area_m2
            conf_area = conf_pixels * pixel_area_m2

            if t > 0:
                prev_cand = (candidate[t-1] == 1) & nogo_mask
                new_candidate = cand_mask & (~prev_cand)

                if np.any(new_candidate):
                    area = np.count_nonzero(new_candidate) * pixel_area_m2
                    log.write(
                        f"[{date}] | Mine {mineid} | LEVEL 1 | CANDIDATE INTRUSION\n"
                        f"Early warning: suspected excavation inside no-go zone.\n"
                        f"Affected area: {area:.1f} m²\n\n"
                    )

            if t > 0:
                prev_conf = (confirmed[t-1] == 1) & nogo_mask
                new_confirmed = conf_mask & (~prev_conf)

                if np.any(new_confirmed):
                    area = np.count_nonzero(new_confirmed) * pixel_area_m2
                    log.write(
                        f"[{date}] | Mine {mineid} | LEVEL 2 | CONFIRMED VIOLATION\n"
                        f"Sustained excavation detected inside no-go zone.\n"
                        f"Affected area: {area:.1f} m²\n\n"
                    )

            if conf_pixels > prev_confirmed_area + expansion_threshold_pixels:
                delta_area = (conf_pixels - prev_confirmed_area) * pixel_area_m2
                log.write(
                    f"[{date}] | Mine {mineid} | LEVEL 3 | VIOLATION EXPANSION\n"
                    f"Confirmed excavation expanding inside no-go zone.\n"
                    f"Newly added area: {delta_area:.1f} m²\n\n"
                )

            prev_confirmed_area = conf_pixels

def NoGoExcavationTimePlot(mineid, outDir, dates, candidate, confirmed, transform,crs, nogo=None):
    if nogo is None:
        return None

    H, W = candidate.shape[1:]
    pixel_area_m2 = abs(transform.a * transform.e)

    nogo = nogo.to_crs(crs)
    nogo_mask = geometry_mask(
        nogo.geometry,
        out_shape=(H, W),
        transform=transform,
        invert=True
    )

    cand_areas = []
    conf_areas = []

    for t in range(len(dates)):
        cand_pixels = np.count_nonzero((candidate[t] == 1) & nogo_mask)
        conf_pixels = np.count_nonzero((confirmed[t] == 1) & nogo_mask)

        cand_areas.append(cand_pixels * pixel_area_m2)
        conf_areas.append(conf_pixels * pixel_area_m2)

    df = pd.DataFrame({
        "date": dates,
        "candidate_area_m2": cand_areas,
        "confirmed_area_m2": conf_areas
    })

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["candidate_area_m2"], label="Candidate (Early)", linestyle="--")
    plt.plot(df["date"], df["confirmed_area_m2"], label="Confirmed", linewidth=2)

    plt.xlabel("Date", fontdict=normalFont)
    plt.ylabel("Excavated Area in No-Go Zone (m²)", fontdict=normalFont)
    plt.title("Excavation Activity Inside No-Go Zone", fontdict=titleFont)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(outDir + f"mine_{mineid}_NoGo_Excavation_vs_Time.png", dpi=300)
    plt.close()

    df.to_csv(outDir + f"mine_{mineid}_NoGo_Excavation_vs_Time.csv", index=False)

    return df

