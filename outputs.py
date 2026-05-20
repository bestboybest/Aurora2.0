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

# Load mine boundary polygons (used only for visualization + area calculations)
gdf = gpd.read_file("./Mine Polygons/SHP/mines_cils.shp").to_crs("EPSG:4326").reset_index(drop=True)
gdf["mine_id"] = gdf.index
gdf = gdf[["mine_id", "area", "perimeter", "geometry"]]

# Shared font styles for all plots (purely cosmetic)
titleFont  = {'weight':'bold', 'color':'orangered', 'size':20, 'name':'Comic Sans MS'}
normalFont = {'color':'maroon', 'size':16}

# -------------------------------------------------------------------
# Converts a binary raster mask into vector polygons.
# Only pixels with value == 1 (excavated) are polygonized.
# Used for overlaying excavation boundaries on RGB imagery.
# -------------------------------------------------------------------
def polygonize(mask, transform, crs):
    geometries = []
    values     = []

    for geom, value in shapes(mask.astype(np.uint8), transform=transform):
        if value == 1:
            geometries.append(shape(geom))
            values.append(value)
    
    exgdf = gpd.GeoDataFrame({"value": values}, geometry=geometries, crs=crs)
    return exgdf

# -------------------------------------------------------------------
# Fetches a quick-look RGB image from Earth Engine as a NumPy array.
# Used only for visualization (not analysis).
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Rasterizes a no-go polygon into a boolean mask aligned to a raster.
# True  -> inside no-go zone
# False -> outside
# -------------------------------------------------------------------
def rasterize_nogo(nogo_gdf, shape, transform):
    return geometry_mask(
        nogo_gdf.geometry,
        out_shape=shape,
        transform=transform,
        invert=True
    )

# -------------------------------------------------------------------
# Debug visualization:
# Shows RGB Sentinel-2 image alongside the raw excavation mask
# at a specific timestep.
# -------------------------------------------------------------------
def debugCluster(mineid, t, dates, E):
    eeDate = ee.Date(dates[t].isoformat())
    mine   = geemap.geopandas_to_ee(gdf[gdf["mine_id"] == mineid])

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(mine)
        .filterDate(eeDate.advance(-10, "day"), eeDate.advance(10, "day"))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
        .median()
        .clip(mine)
    )

    rgb = eeToNumpy(
        s2,
        region=mine.geometry(),
        vis={"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
    )

    # Mask non-excavated pixels for clarity
    raw_exc = np.ma.masked_where(E[t] != 1, E[t])

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

# -------------------------------------------------------------------
# Generates paired spatial maps:
# 1) RGB image + excavation boundary
# 2) Pixel-wise excavation confidence map
# Shown at representative percentiles of the timeline.
# -------------------------------------------------------------------
def makeSpatialMaps(mineid, dir, dates, retroConfirmed, confidence, transform, crs, nogo=None):
    T = len(dates)

    percentiles = [0, 25, 50, 75, 100]
    t_idxs = [int(round(p / 100 * (T - 1))) for p in percentiles]

    retroPolys = [
        polygonize(retroConfirmed[t], transform, crs)
        for t in t_idxs
    ]

    for i, (t, p) in enumerate(zip(t_idxs, percentiles)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        mine   = geemap.geopandas_to_ee(gdf[gdf["mine_id"] == mineid])
        eeDate = ee.Date(dates[t].isoformat())
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(mine)
            .filterDate(eeDate.advance(-10, "day"), eeDate.advance(10, "day"))
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
            .median()
            .clip(mine)
        )

        rgb = eeToNumpy(
            s2,
            region=mine.geometry(),
            vis={"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
        )

        minx, miny, maxx, maxy = gdf[gdf["mine_id"] == mineid].total_bounds
        extent = [minx, maxx, miny, maxy]

        ax1.imshow(rgb, extent=extent, origin="upper", alpha=0.85)
        ax1.set_aspect("equal")

        poly = retroPolys[i].to_crs("EPSG:4326")
        if not poly.empty and poly.is_valid.any():
            poly.boundary.plot(ax=ax1, edgecolor="cyan", linewidth=2, zorder=10)

        gdf[gdf["mine_id"] == mineid].boundary.plot(ax=ax1, edgecolor="black", linewidth=2)

        if nogo is not None:
            nogo = nogo.to_crs("EPSG:4326")
            nogo.boundary.plot(ax=ax1, edgecolor="yellow", linewidth=2, linestyle="--")

        ax1.set_title(f"Retroconfirmed Excavated Area – {dates[t].isoformat()}", fontdict=titleFont)
        ax1.axis("off")

        # --- Confidence heatmap ---
        conf        = confidence[t]
        conf_masked = np.ma.masked_where(conf == 0, conf)

        minegdf = gdf[gdf["mine_id"] == mineid].to_crs(crs)
        H, W    = conf_masked.shape
        bounds  = array_bounds(H, W, transform)
        extent  = [bounds[0], bounds[2], bounds[1], bounds[3]]

        mine_mask = geometry_mask(
            minegdf.geometry,
            out_shape=conf_masked.shape,
            transform=transform,
            invert=True
        )

        conf_vis = np.ma.masked_where(~mine_mask, conf_masked)

        im = ax2.imshow(
            conf_vis,
            cmap="RdYlGn_r",
            vmin=0, vmax=1,
            extent=extent,
            origin="upper"
        )

        ax2.set_title(f"Confidence on Excavation – {dates[t].isoformat()}", fontdict=titleFont)
        minegdf.boundary.plot(ax=ax2, edgecolor="black", linewidth=2)
        ax2.axis("off")
        plt.colorbar(im, ax=ax2, fraction=0.046)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(
            dir + f"mine_{mineid}_spatialMap_{p}percent.png",
            dpi=300, bbox_inches="tight", pad_inches=0.2
        )
        plt.close()

# -------------------------------------------------------------------
# Creates a grid of spatial snapshots (0–100% timeline)
# showing how excavation evolves over time.
# -------------------------------------------------------------------
def retroConfirmedAnalysis(mineid, dates, dir, retroConfirmed, transform, crs, nogo=None):
    T = len(dates)

    percentiles = list(range(0, 101, 10))
    t_idxs      = [int(round(p / 100 * (T - 1))) for p in percentiles]

    minegdf = gdf[gdf["mine_id"] == mineid]
    mine    = geemap.geopandas_to_ee(minegdf)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    minx, miny, maxx, maxy = minegdf.total_bounds
    extent = [minx, maxx, miny, maxy]

    for ax, t, p in zip(axes, t_idxs, percentiles):
        date   = dates[t]
        eeDate = ee.Date(date.isoformat())
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(mine)
            .filterDate(eeDate.advance(-10, "day"), eeDate.advance(10, "day"))
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
            .median()
            .clip(mine)
        )

        rgb = eeToNumpy(
            s2,
            region=mine.geometry(),
            vis={"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
        )

        ax.imshow(rgb, extent=extent, origin="upper", alpha=0.85)

        poly = polygonize(retroConfirmed[t], transform, crs).to_crs("EPSG:4326")
        if not poly.empty and poly.is_valid.any():
            poly.boundary.plot(ax=ax, edgecolor="cyan", linewidth=2, zorder=10)

        minegdf.boundary.plot(ax=ax, edgecolor="black", linewidth=2)

        if nogo is not None:
            nogo = nogo.to_crs("EPSG:4326")
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

# -------------------------------------------------------------------
# Computes total retroconfirmed excavated area over time.
# -------------------------------------------------------------------
def ExcavationTimePlot(mineid, dir, retroConfirmed, dates, transform):
    pixel_area_m2 = abs(transform.a * transform.e)

    areas = []
    for t in range(len(dates)):
        excavated_pixels = np.count_nonzero(retroConfirmed[t] == 1)
        areas.append(excavated_pixels * pixel_area_m2)

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "excavated_area_m2": areas
    })

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["excavated_area_m2"], linewidth=2)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))

    ax.set_xlabel("Year", fontdict=normalFont)
    ax.set_ylabel("Excavated Area (m²)", fontdict=normalFont)
    ax.set_title("Retroconfirmed Excavated Area vs Time (2018–2025)", fontdict=titleFont)

    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    plt.tight_layout()
    plt.savefig(dir + f"mine_{mineid}_AreavsTime.png", dpi=300)
    plt.close()

    return df

# -------------------------------------------------------------------
# Same as above, but for *candidate* (unconfirmed) excavation.
# -------------------------------------------------------------------
def CandidateExcavationTimePlot(mineid, dir, candidate, dates, transform):
    pixel_area_m2 = abs(transform.a * transform.e)

    areas = []
    for t in range(len(dates)):
        candidate_pixels = np.count_nonzero(candidate[t] == 1)
        areas.append(candidate_pixels * pixel_area_m2)

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "candidate_excavated_area_m2": areas
    })

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["candidate_excavated_area_m2"], linewidth=2, color="orange")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))

    ax.set_xlabel("Year", fontdict=normalFont)
    ax.set_ylabel("Candidate Excavated Area (m²)", fontdict=normalFont)
    ax.set_title("Candidate Excavated Area vs Time (Unconfirmed Signals)", fontdict=titleFont)

    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    plt.tight_layout()
    plt.savefig(dir + f"mine_{mineid}_CandidateAreavsTime.png", dpi=300)
    plt.close()

    return df

# -------------------------------------------------------------------
# Computes excavation growth rate (m²/day).
# -------------------------------------------------------------------
def GrowthRatePlot(mineid, dir, retroConfirmed, dates, transform):
    pixel_area_m2 = abs(transform.a * transform.e)

    areas = []
    for t in range(len(dates)):
        excavated_pixels = np.count_nonzero(retroConfirmed[t] == 1)
        areas.append(excavated_pixels * pixel_area_m2)

    rates = [0.0]
    for i in range(1, len(areas)):
        delta_days = (dates[i] - dates[i - 1]).days
        rates.append(0.0 if delta_days == 0 else (areas[i] - areas[i - 1]) / delta_days)

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

# -------------------------------------------------------------------
# Side-by-side comparison of candidate vs retroconfirmed excavation.
# -------------------------------------------------------------------
def ComparisionPlot(mineid, dir, retro_df, candidate_df):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        retro_df["date"],
        retro_df["excavated_area_m2"],
        linewidth=3, color="tab:blue",
        label="Retroconfirmed Excavation"
    )
    ax.plot(
        candidate_df["date"],
        candidate_df["candidate_excavated_area_m2"],
        linewidth=2, linestyle="--", color="tab:orange",
        label="Candidate Excavation"
    )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))

    ax.set_xlabel("Year", fontdict=normalFont)
    ax.set_ylabel("Excavated Area (m²)", fontdict=normalFont)
    ax.set_title("Candidate vs Retroconfirmed Excavated Area", fontdict=titleFont)

    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)
    ax.legend()

    plt.tight_layout()
    plt.savefig(dir + f"mine_{mineid}_ComparisionPlot.png", dpi=300)
    plt.close()

# -------------------------------------------------------------------
# Normalizes excavated area by total mine area.
# Enables comparison across mines of different sizes.
# -------------------------------------------------------------------
def NormalizedExcavationPlot(mineid, dir, retro_df):
    mine_area_m2 = (
        gdf[gdf["mine_id"] == mineid]
        .to_crs("EPSG:6933")
        .geometry.area.values[0]
    )

    df = retro_df.copy()
    df["normalized_excavation"] = df["excavated_area_m2"] / mine_area_m2

    out_path = dir + f"mine_{mineid}_ExcavationIntensity.csv"
    df.to_csv(out_path, index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["normalized_excavation"], linewidth=3, color="tab:green")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Year", fontdict=normalFont)
    ax.set_ylabel("Normalized Excavation (fraction of mine)", fontdict=normalFont)
    ax.set_title("Normalized Retroconfirmed Excavation vs Time", fontdict=titleFont)

    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    plt.tight_layout()
    plt.savefig(dir + f"mine_{mineid}_NormalizedExcavation.png", dpi=300)
    plt.close()

    return df

# -------------------------------------------------------------------
# Generates a spatial map showing when excavation was first confirmed.
# Color encodes days since monitoring start.
# -------------------------------------------------------------------
def FirstSeenDateMap(mineid, dir, confirmedFirstSeen, dates, transform, crs, nogo=None):
    H, W = confirmedFirstSeen.shape

    start_date          = dates[0]
    first_detection_days = np.full((H, W), np.nan, dtype=np.float32)

    for y in range(H):
        for x in range(W):
            t0 = confirmedFirstSeen[y, x]
            if t0 >= 0:
                first_detection_days[y, x] = (dates[t0] - start_date).days

    detection_masked = np.ma.masked_invalid(first_detection_days)

    fig, ax = plt.subplots(figsize=(5, 5))

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
        fontdict=titleFont, pad=12
    )
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Days Since Monitoring Start", fontdict=normalFont)

    plt.savefig(
        dir + f"mine_{mineid}_FirstSeenPlot.png",
        dpi=300, bbox_inches="tight", pad_inches=0.2
    )
    plt.close()

    return first_detection_days

# -------------------------------------------------------------------
# No-Go Zone Alert System
# -------------------------------------------------------------------
def NoGoAlertSystem(
    mineid, outDir, dates,
    candidate, confirmed,
    transform, crs,
    nogo=None,
    expansion_threshold_pixels=20
):
    if nogo is None:
        return

    alert_path    = outDir + f"mine_{mineid}_alerts.log"
    H, W          = candidate.shape[1:]
    pixel_area_m2 = abs(transform.a * transform.e)

    nogo     = nogo.to_crs(crs)
    nogo_mask = rasterize_nogo(nogo, (H, W), transform)

    with open(alert_path, "a") as log:
        prev_level3_triggered = False
        for t in range(len(dates)):
            date = dates[t]

            cand_mask  = (candidate[t] == 1) & nogo_mask
            conf_mask  = (confirmed[t] == 1) & nogo_mask
            conf_pixels = np.count_nonzero(conf_mask)

            if t > 0:
                prev_cand    = (candidate[t-1] == 1) & nogo_mask
                new_candidate = cand_mask & (~prev_cand)

                if np.any(new_candidate):
                    area = np.count_nonzero(new_candidate) * pixel_area_m2
                    log.write(
                        f"[{date}] | Mine {mineid} | LEVEL 1 | CANDIDATE INTRUSION\n"
                        f"Early warning: suspected excavation inside no-go zone.\n"
                        f"Affected area: {area:.1f} m²\n\n"
                    )

            if t > 0:
                prev_conf    = (confirmed[t-1] == 1) & nogo_mask
                new_confirmed = conf_mask & (~prev_conf)

                if np.any(new_confirmed):
                    area = np.count_nonzero(new_confirmed) * pixel_area_m2
                    log.write(
                        f"[{date}] | Mine {mineid} | LEVEL 2 | CONFIRMED VIOLATION\n"
                        f"Sustained excavation detected inside no-go zone.\n"
                        f"Affected area: {area:.1f} m²\n\n"
                    )

            window_days  = 60
            window_start = t
            while window_start > 0 and (dates[t] - dates[window_start]).days < window_days:
                window_start -= 1

            prev_conf_mask   = (confirmed[window_start] == 1) & nogo_mask
            prev_conf_pixels = np.count_nonzero(prev_conf_mask)
            level3_condition = conf_pixels > prev_conf_pixels + expansion_threshold_pixels

            if level3_condition and not prev_level3_triggered:
                delta_pixels = conf_pixels - prev_conf_pixels
                delta_area   = delta_pixels * pixel_area_m2
                log.write(
                    f"[{date}] | Mine {mineid} | LEVEL 3 | VIOLATION EXPANSION\n"
                    f"Sustained expansion of confirmed excavation inside no-go zone.\n"
                    f"Added area over last {window_days} days: {delta_area:.1f} m²\n\n"
                )

            prev_level3_triggered = level3_condition

# -------------------------------------------------------------------
# Time-series plot of excavation activity inside no-go zones.
# -------------------------------------------------------------------
def NoGoExcavationTimePlot(
    mineid, outDir, dates,
    candidate, confirmed,
    transform, crs,
    nogo=None
):
    if nogo is None:
        return None

    H, W          = candidate.shape[1:]
    pixel_area_m2 = abs(transform.a * transform.e)

    nogo      = nogo.to_crs(crs)
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
        "date":               dates,
        "candidate_area_m2":  cand_areas,
        "confirmed_area_m2":  conf_areas
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

# -------------------------------------------------------------------
# SAR Coverage Diagnostic Plot
#
# Shows three time-series on one plot:
#   1. Fraction of mine pixels with valid S2 data (not cloud-masked)
#   2. Fraction filled by SAR gap-fill
#   3. Fraction still unknown after both sources
#   (all fractions are relative to total mine pixels)
#
# Inputs:
#   Efixed_before : Efixed BEFORE sarGapFill() (raw S2 only)
#   Efixed_after  : Efixed AFTER  sarGapFill() (S2 + SAR)
#   Both have shape (T, H, W) with values 0/1/2
# -------------------------------------------------------------------
def SARCoveragePlot(mineid, outDir, dates, Efixed_before, Efixed_after):
    T = len(dates)
    total_pixels = Efixed_before.shape[1] * Efixed_before.shape[2]

    s2_valid_frac   = []
    sar_filled_frac = []
    still_unknown   = []

    for t in range(T):
        before_unknown = (Efixed_before[t] == 2)
        after_unknown  = (Efixed_after[t]  == 2)

        n_s2_valid  = total_pixels - np.count_nonzero(before_unknown)
        n_sar_filled = np.count_nonzero(before_unknown & ~after_unknown)
        n_unknown    = np.count_nonzero(after_unknown)

        s2_valid_frac.append(n_s2_valid   / total_pixels)
        sar_filled_frac.append(n_sar_filled / total_pixels)
        still_unknown.append(n_unknown    / total_pixels)

    df = pd.DataFrame({
        "date":             pd.to_datetime(dates),
        "s2_valid_frac":    s2_valid_frac,
        "sar_filled_frac":  sar_filled_frac,
        "still_unknown":    still_unknown
    })

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(df["date"], df["s2_valid_frac"],
                    alpha=0.4, color="tab:blue",  label="S2 valid coverage")
    ax.fill_between(df["date"], df["sar_filled_frac"],
                    alpha=0.4, color="tab:green", label="SAR gap-fill")
    ax.fill_between(df["date"], df["still_unknown"],
                    alpha=0.4, color="tab:red",   label="Still unknown")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax.set_ylim(0, 1)
    ax.set_xlabel("Date", fontdict=normalFont)
    ax.set_ylabel("Fraction of Mine Pixels", fontdict=normalFont)
    ax.set_title("S2 Coverage vs SAR Gap-Fill vs Unknown", fontdict=titleFont)

    ax.grid(alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(outDir + f"mine_{mineid}_SARCoverage.png", dpi=300)
    plt.close()

    df.to_csv(outDir + f"mine_{mineid}_SARCoverage.csv", index=False)

    return df

# -------------------------------------------------------------------
# SAR Histogram Analysis
#
# Loads the CSV produced by sarAnalysisStart() and generates a
# diagnostic figure with two sections:
#
#   TOP ROW  — overlaid KDE + histogram for each feature
#              (VV, VH, VV/VH) with all three classes on one axis.
#              Makes it immediately obvious whether distributions
#              separate or overlap.
#
#   BOTTOM ROW — same data as box-and-whisker plots with individual
#                data points (stripplot-style) overlaid, so you can
#                see spread, outliers, and the current fixed threshold.
#
# Classes:
#   excavated     → confirmed S2 label == 1  (red)
#   non-excavated → confirmed S2 label == 0  (green)
#   cloud-masked  → S2 label == 2            (grey, dashed — reference only)
#
# Annotations printed on each histogram panel:
#   • median per class
#   • Cohen's d between excavated and non-excavated
#     (effect size — >0.8 = well separated, <0.5 = poor separation)
#   • current fixed threshold line (VV only: -12 dB)
#   • mine-specific threshold line if vv_thresh_val is passed in
#
# Statistics summary CSV is also written alongside the figure.
#
# Parameters:
#   mineid         : mine index
#   outDir         : output directory (same as other plots)
#   debug          : suffix matching sarAnalysisStart() output
#   vv_thresh_val  : optional mine-specific VV threshold (dB) to
#                    annotate on the VV panels. Pass the value
#                    printed by sarMonitoringStart() to compare
#                    against the old fixed -12 dB rule.
#   delta_thresh_val: optional mine-specific delta threshold (dB)
#                    to annotate on the VV_VH_ratio panel.
# -------------------------------------------------------------------
def SARHistogramPlot(mineid, outDir, debug=0, vv_thresh_val=None, delta_thresh_val=None):
    import scipy.stats as stats

    mainDir  = f"./Mine Data/Mine_{mineid}_data/"
    csv_path = mainDir + f"mine_{mineid}_sar_analysis_{debug}.csv"

    df = pd.read_csv(csv_path).dropna(subset=["VV", "VH", "VV_VH_ratio"])

    SAR_FEATURES = ["VV", "VH", "VV_VH_ratio"]
    CLASSES      = ["excavated", "non-excavated", "cloud-masked"]
    COLORS       = {
        "excavated":     "#d62728",
        "non-excavated": "#2ca02c",
        "cloud-masked":  "#aaaaaa",
    }
    XLABELS = {
        "VV":          "VV backscatter (dB)",
        "VH":          "VH backscatter (dB)",
        "VV_VH_ratio": "VV − VH (dB, log ratio)",
    }

    # ---- helpers -------------------------------------------------
    def cohens_d(a, b):
        """Effect size between two 1-D arrays."""
        if len(a) < 2 or len(b) < 2:
            return np.nan
        pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
        return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan

    def kde_curve(values, ax, color, linestyle="-"):
        """Overlay a smooth KDE on top of the histogram."""
        if len(values) < 5:
            return
        kde  = stats.gaussian_kde(values, bw_method="silverman")
        xmin, xmax = ax.get_xlim()
        xs   = np.linspace(xmin, xmax, 300)
        ys   = kde(xs)
        # Scale KDE to match histogram height
        ax2  = ax.twinx()
        ax2.plot(xs, ys, color=color, linewidth=2, linestyle=linestyle, alpha=0.9)
        ax2.set_yticks([])
        ax2.set_ylim(bottom=0)

    # ---- figure layout -------------------------------------------
    fig, axes = plt.subplots(
        nrows=2, ncols=3,
        figsize=(16, 10),
        gridspec_kw={"height_ratios": [1.6, 1]}
    )
    fig.suptitle(
        f"Mine {mineid} — SAR Feature Distributions by S2 Pixel Label",
        fontsize=14, fontweight="bold", y=1.01
    )

    stats_rows = []

    # ---- TOP ROW: overlaid histograms + KDE ----------------------
    for col, feat in enumerate(SAR_FEATURES):
        ax = axes[0][col]

        for cls in CLASSES:
            vals = df[df["pixel_label"] == cls][feat].dropna().values
            if len(vals) == 0:
                continue
            color     = COLORS[cls]
            linestyle = "--" if cls == "cloud-masked" else "-"
            alpha     = 0.35 if cls == "cloud-masked" else 0.55

            ax.hist(
                vals, bins=60,
                color=color, alpha=alpha,
                label=f"{cls} (n={len(vals):,})",
                density=True, edgecolor="none"
            )
            # Median tick
            med = np.median(vals)
            ax.axvline(med, color=color, linewidth=1.8,
                       linestyle=":", alpha=0.9)
            ax.text(med, ax.get_ylim()[1] * 0.01, f"{med:.1f}",
                    color=color, fontsize=7, rotation=90,
                    va="bottom", ha="right")

        # KDE overlays (drawn after so xlim is stable)
        for cls in CLASSES:
            vals = df[df["pixel_label"] == cls][feat].dropna().values
            kde_curve(vals, ax, COLORS[cls],
                      "--" if cls == "cloud-masked" else "-")

        # Fixed threshold reference (VV only)
        if feat == "VV":
            ax.axvline(-12, color="orange", linewidth=1.5,
                       linestyle="-.", label="fixed −12 dB", zorder=5)
            if vv_thresh_val is not None:
                ax.axvline(vv_thresh_val, color="gold", linewidth=1.5,
                           linestyle="-.", label=f"mine thresh {vv_thresh_val:.1f} dB", zorder=5)

        if feat == "VV_VH_ratio" and delta_thresh_val is not None:
            ax.axvline(delta_thresh_val, color="gold", linewidth=1.5,
                       linestyle="-.", label=f"delta thresh +{delta_thresh_val:.1f} dB", zorder=5)

        # Cohen's d annotation
        ex_vals  = df[df["pixel_label"] == "excavated"][feat].dropna().values
        non_vals = df[df["pixel_label"] == "non-excavated"][feat].dropna().values
        d        = cohens_d(ex_vals, non_vals)
        sep_label = (
            "well separated" if abs(d) > 0.8
            else "moderate" if abs(d) > 0.5
            else "poor separation"
        )
        ax.set_title(
            f"{feat}\nCohen's d = {d:.2f}  ({sep_label})",
            fontsize=10, fontweight="bold"
        )
        ax.set_xlabel(XLABELS[feat], fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

        # Collect stats
        for cls in CLASSES:
            vals = df[df["pixel_label"] == cls][feat].dropna().values
            stats_rows.append({
                "feature": feat, "class": cls,
                "n": len(vals),
                "median": round(np.median(vals), 3) if len(vals) else np.nan,
                "mean":   round(np.mean(vals),   3) if len(vals) else np.nan,
                "std":    round(np.std(vals),     3) if len(vals) else np.nan,
                "p5":     round(np.percentile(vals,  5), 3) if len(vals) else np.nan,
                "p95":    round(np.percentile(vals, 95), 3) if len(vals) else np.nan,
                "cohens_d_vs_nonex": round(d, 3) if cls == "excavated" else np.nan,
            })

    # ---- BOTTOM ROW: box plots -----------------------------------
    for col, feat in enumerate(SAR_FEATURES):
        ax = axes[1][col]

        plot_data   = []
        plot_labels = []
        plot_colors = []

        for cls in CLASSES:
            vals = df[df["pixel_label"] == cls][feat].dropna().values
            if len(vals) == 0:
                continue
            plot_data.append(vals)
            plot_labels.append(cls.replace("-", "-\n"))
            plot_colors.append(COLORS[cls])

        bp = ax.boxplot(
            plot_data,
            labels=plot_labels,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
            widths=0.5
        )
        for patch, color in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

        # Strip plot: jittered individual points (capped at 300/class for speed)
        for i, (vals, color) in enumerate(zip(plot_data, plot_colors), start=1):
            sample = vals if len(vals) <= 300 else np.random.choice(vals, 300, replace=False)
            jitter = np.random.uniform(-0.18, 0.18, size=len(sample))
            ax.scatter(i + jitter, sample, color=color, alpha=0.18,
                       s=4, zorder=3, linewidths=0)

        # Threshold reference lines
        if feat == "VV":
            ax.axhline(-12, color="orange", linewidth=1.4, linestyle="-.",
                       label="fixed −12 dB")
            if vv_thresh_val is not None:
                ax.axhline(vv_thresh_val, color="gold", linewidth=1.4,
                           linestyle="-.", label=f"mine thresh {vv_thresh_val:.1f}")
            ax.legend(fontsize=7)

        ax.set_ylabel(XLABELS[feat], fontsize=9)
        ax.set_title(f"{feat} — box + strip", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out_png = outDir + f"mine_{mineid}_SARHistograms_{debug}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_png}")

    # ---- Stats CSV -----------------------------------------------
    stats_df = pd.DataFrame(stats_rows)
    out_csv  = outDir + f"mine_{mineid}_SARHistograms_{debug}_stats.csv"
    stats_df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    # ---- Console summary -----------------------------------------
    print(f"\n{'Feature':<14} {'Class':<18} {'Median':>7} {'Mean':>7} "
          f"{'Std':>6} {'P5':>7} {'P95':>7} {'N':>7} {'Cohen d':>9}")
    print("─" * 86)
    for feat in SAR_FEATURES:
        for cls in CLASSES:
            row = stats_df[(stats_df["feature"] == feat) &
                           (stats_df["class"]   == cls)]
            if row.empty or row["n"].values[0] == 0:
                continue
            r = row.iloc[0]
            d_str = f"{r['cohens_d_vs_nonex']:>9.2f}" if not np.isnan(r["cohens_d_vs_nonex"]) else "         —"
            print(f"{feat:<14} {cls:<18} {r['median']:>7.2f} {r['mean']:>7.2f} "
                  f"{r['std']:>6.2f} {r['p5']:>7.2f} {r['p95']:>7.2f} "
                  f"{int(r['n']):>7,}{d_str}")
        print()

    return stats_df


def KSelectionPlot(mineid, dir, k_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_scores.keys()), list(k_scores.values()), marker='o')

    plt.xlabel("K", fontdict=normalFont)
    plt.ylabel("Silhouette Score", fontdict=normalFont)
    plt.title("K Selection using Silhouette Score", fontdict=titleFont)

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(dir + f"mine_{mineid}_KSelection.png", dpi=300)
    plt.close()