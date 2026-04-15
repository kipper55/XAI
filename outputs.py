


# outputs.py
# Step 4: Explainability Report, CSV Export, and Interactive GUI Visualisation

import os
import numpy as np
import pandas as pd
import rioxarray
import geopandas as gpd
from pyproj import Transformer
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import CheckButtons
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
DATASET_DIR   = r"C:\Users\antvi\PYTHON_PROJECTS\SAREnv\sarenv_dataset\ben_nevis"
TIF_PATH      = os.path.join(DATASET_DIR, "ben_nevis_geo.tif")
PATH_HEATMAP  = os.path.join(DATASET_DIR, "heatmap.npy")
PATH_SLOPE_P  = os.path.join(DATASET_DIR, "slope_p.npy")
PATH_DIST_P   = os.path.join(DATASET_DIR, "dist_p.npy")
PATH_FEATURES = os.path.join(DATASET_DIR, "features.geojson")
OUTPUT_CSV    = os.path.join(DATASET_DIR, "survivor_explainability.csv")

NUM_SURVIVORS = 10_000
RESOLUTION_M  = 30

FEATURE_PROBABILITIES = {
    "linear": 0.25, "field": 0.14, "structure": 0.13, "road": 0.13,
    "drainage": 0.12, "water": 0.08, "woodland": 0.07, "rock": 0.04,
    "scrub": 0.03, "brush": 0.02
}

# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_all():
    rds       = rioxarray.open_rasterio(TIF_PATH)
    elevation = rds.values[0] if rds.values.ndim == 3 else rds.values
    heatmap   = np.load(PATH_HEATMAP)
    slope_p   = np.load(PATH_SLOPE_P)
    dist_p    = np.load(PATH_DIST_P)
    bounds    = rds.rio.bounds()
    crs_str   = rds.rio.crs.to_string()

    try:
        gdf = gpd.read_file(PATH_FEATURES).to_crs("EPSG:4326")
    except Exception:
        gdf = None

    return elevation, heatmap, slope_p, dist_p, bounds, crs_str, gdf

# =============================================================================
# 2. PIXEL <-> LAT/LON HELPERS
# =============================================================================
def build_transformer(bounds, crs_str, shape):
    rows, cols = shape
    left, bottom, right, top = bounds
    px_w = (right - left) / cols
    px_h = (top - bottom) / rows
    transformer = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)

    def px_to_latlon(row, col):
        x = left + (col + 0.5) * px_w
        y = top  - (row + 0.5) * px_h
        lon, lat = transformer.transform(x, y)
        return lat, lon

    return px_to_latlon

# =============================================================================
# 3. SIMULATE SURVIVORS USING OPTIMISED WEIGHTS
# =============================================================================
def simulate_survivors(elevation, heatmap, slope_p, dist_p, bounds, crs_str, theta_hat):
    rows, cols = elevation.shape
    center_row = rows // 2
    center_col = cols // 2
    lkp_elev   = elevation[center_row, center_col]

    feature_map = heatmap / heatmap.max()

    # Build POC using optimised weights from subject_optimise.py
    poc = (
        slope_p     * theta_hat["slope"] +
        dist_p      * theta_hat["dist"]  +
        feature_map * theta_hat["feat"]
    )
    poc = np.clip(poc, 1e-12, None)
    poc /= poc.sum()

    flat_poc           = poc.ravel()
    indices            = np.random.choice(len(flat_poc), size=NUM_SURVIVORS, p=flat_poc)
    end_rows, end_cols = np.unravel_index(indices, (rows, cols))
    px_to_latlon       = build_transformer(bounds, crs_str, (rows, cols))

    records = []
    for i, idx in enumerate(indices):
        er, ec = end_rows[i], end_cols[i]

        s_val  = slope_p.ravel()[idx]
        d_val  = dist_p.ravel()[idx]
        f_val  = feature_map.ravel()[idx]
        total  = s_val + d_val + f_val + 1e-12

        slope_pct = s_val / total
        dist_pct  = d_val / total
        feat_pct  = f_val / total

        drivers    = {"slope": slope_pct, "distance": dist_pct, "feature": feat_pct}
        top_driver = max(drivers, key=drivers.get)

        dist_px  = np.sqrt((ec - center_col)**2 + (er - center_row)**2)
        dist_m   = dist_px * RESOLUTION_M
        end_elev = elevation[er, ec]
        gravity  = end_elev < lkp_elev

        feat_labels  = list(FEATURE_PROBABILITIES.keys())
        feat_weights = np.array(list(FEATURE_PROBABILITIES.values()))
        feat_label   = np.random.choice(feat_labels, p=feat_weights / feat_weights.sum())

        lat, lon         = px_to_latlon(er, ec)
        lkp_lat, lkp_lon = px_to_latlon(center_row, center_col)

        records.append({
            "survivor_id":            i,
            "lkp_latitude":           round(lkp_lat, 6),
            "lkp_longitude":          round(lkp_lon, 6),
            "lsf_latitude":           round(lat, 6),
            "lsf_longitude":          round(lon, 6),
            "lsf_row":                er,
            "lsf_col":                ec,
            "distance_from_lkp_m":    round(dist_m, 1),
            "lkp_elevation_m":        round(float(lkp_elev), 1),
            "lsf_elevation_m":        round(float(end_elev), 1),
            "elevation_delta_m":      round(float(end_elev - lkp_elev), 1),
            "gravity_bias":           gravity,
            "slope_pct":              round(slope_pct, 4),
            "distance_pct":           round(dist_pct, 4),
            "feature_pct":            round(feat_pct, 4),
            "top_logit_driver":       top_driver,
            "specific_feature_label": feat_label,
            "poc_value":              round(float(flat_poc[idx]), 8),
            "weight_slope":           round(theta_hat["slope"], 4),
            "weight_dist":            round(theta_hat["dist"],  4),
            "weight_feat":            round(theta_hat["feat"],  4),
        })

    df = pd.DataFrame(records)
    return df, poc, (center_row, center_col), lkp_elev

# =============================================================================
# 4. CONSOLE EXPLAINABILITY REPORT
# =============================================================================
def print_explainability(df, lkp_elev, theta_hat):
    total         = len(df)
    mean_radius   = df["distance_from_lkp_m"].mean()
    gravity_pct   = df["gravity_bias"].mean() * 100
    max_radius    = df["distance_from_lkp_m"].max()
    median_radius = df["distance_from_lkp_m"].median()

    top_driver_counts = df["top_logit_driver"].value_counts()
    feat_label_counts = df["specific_feature_label"].value_counts()

    print("\n" + "═" * 70)
    print("   GLOBAL EXPLAINABILITY REPORT — BEN NEVIS SAR")
    print("═" * 70)
    print(f"  Total subjects simulated:     {total:,}")
    print(f"  LKP elevation:                {lkp_elev:.1f} m")
    print()
    print("  OPTIMISED WEIGHTS USED")
    print(f"    slope:    {theta_hat['slope']:.4f}")
    print(f"    distance: {theta_hat['dist']:.4f}")
    print(f"    feature:  {theta_hat['feat']:.4f}")
    print()
    print("  SEARCH RADIUS")
    print(f"    Mean distance from LKP:     {mean_radius:.1f} m")
    print(f"    Median distance from LKP:   {median_radius:.1f} m")
    print(f"    Max distance from LKP:      {max_radius:.1f} m")
    print()
    print("  GRAVITY BIAS")
    print(f"    Subjects found downhill:    {gravity_pct:.1f}%")
    print(f"    Subjects found uphill:      {100 - gravity_pct:.1f}%")
    print()
    print("  TOP LOGIT DRIVER")
    for driver, count in top_driver_counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {driver:<12s}  {count:>6,}  ({pct:5.1f}%)  {bar}")
    print()
    print("  FEATURE LABEL INFLUENCE")
    for label, count in feat_label_counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {label:<12s}  {count:>6,}  ({pct:5.1f}%)  {bar}")
    print()
    print("  ELEVATION DELTA STATS")
    print(f"    Mean Δ elevation:           {df['elevation_delta_m'].mean():.1f} m")
    print(f"    Std  Δ elevation:           {df['elevation_delta_m'].std():.1f} m")
    print()
    print("  WEIGHT CONTRIBUTION AVERAGES")
    print(f"    Slope influence:            {df['slope_pct'].mean()*100:.1f}%")
    print(f"    Distance influence:         {df['distance_pct'].mean()*100:.1f}%")
    print(f"    Feature influence:          {df['feature_pct'].mean()*100:.1f}%")
    print("═" * 70 + "\n")

# =============================================================================
# 5. INTERACTIVE GUI
# =============================================================================
def launch_gui(df, elevation, heatmap, slope_p, dist_p, poc, lkp_coords,
               gdf, bounds, crs_str, theta_hat):
    rows, cols             = elevation.shape
    center_row, center_col = lkp_coords

    elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    poc_norm  = poc / poc.max()

    # ── Figure & axes ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 10), facecolor="#0d1117")
    fig.canvas.manager.set_window_title("SAR Explainability — Ben Nevis")

    w  = theta_hat
    ax = fig.add_axes([0.18, 0.05, 0.78, 0.90])
    ax.set_facecolor("#0d1117")
    ax.set_title(
        f"BEN NEVIS — SAR  │  optimised weights:  "
        f"slope={w['slope']:.3f}   dist={w['dist']:.3f}   feat={w['feat']:.3f}",
        color="#e6edf3", fontsize=11, fontweight="bold",
        fontfamily="monospace", pad=10
    )
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Base hillshade ────────────────────────────────────────────────────────
    from matplotlib.colors import LightSource
    ls        = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(elevation, vert_exag=3, dx=RESOLUTION_M, dy=RESOLUTION_M)
    ax.imshow(hillshade, cmap="gray", alpha=0.5, origin="upper",
              extent=[0, cols, rows, 0])

    # ── Raster toggle layers ──────────────────────────────────────────────────
    im_elev = ax.imshow(elev_norm, cmap="terrain", alpha=0.55,
                        origin="upper", extent=[0, cols, rows, 0], visible=False)

    im_poc  = ax.imshow(poc_norm, cmap="hot", alpha=0.65,
                        origin="upper", extent=[0, cols, rows, 0], visible=False)

    # ── OSM vector features ───────────────────────────────────────────────────
    FEATURE_COLOURS = {
        "woodland":  "#2d6a2d",
        "water":     "#1a6fa8",
        "road":      "#c8a84b",
        "linear":    "#a0522d",
        "drainage":  "#4a90d9",
        "field":     "#8fbc6a",
        "structure": "#c0392b",
        "rock":      "#7f8c8d",
        "scrub":     "#6aab6a",
        "brush":     "#9b7653",
    }

    osm_artists = []
    if gdf is not None and not gdf.empty:
        left, bottom, right, top_b = bounds
        transformer_to_crs = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
        px_w = (right - left)   / cols
        px_h = (top_b - bottom) / rows

        def latlon_to_px(lat, lon):
            x, y   = transformer_to_crs.transform(lon, lat)
            px_col = (x - left)   / px_w
            px_row = (top_b - y)  / px_h
            return px_col, px_row

        print("--- Projecting OSM features to pixel space ---")
        for f_type, colour in FEATURE_COLOURS.items():
            subset = gdf[gdf["feature_type"] == f_type]
            if subset.empty:
                continue
            for geom in subset.geometry:
                if geom is None or geom.is_empty:
                    continue
                try:
                    gtype = geom.geom_type
                    if gtype == "Point":
                        px_col, px_row = latlon_to_px(geom.y, geom.x)
                        art, = ax.plot(px_col, px_row, "o", color=colour,
                                       markersize=3, alpha=0.7, zorder=3, visible=False)
                        osm_artists.append(art)
                    elif gtype in ("LineString", "MultiLineString"):
                        lines = [geom] if gtype == "LineString" else list(geom.geoms)
                        for line in lines:
                            xs, ys = zip(*[latlon_to_px(lat, lon)
                                           for lon, lat in line.coords])
                            art, = ax.plot(xs, ys, color=colour, linewidth=0.9,
                                           alpha=0.8, zorder=3, visible=False)
                            osm_artists.append(art)
                    elif gtype in ("Polygon", "MultiPolygon"):
                        polys = [geom] if gtype == "Polygon" else list(geom.geoms)
                        for poly in polys:
                            xs, ys = zip(*[latlon_to_px(lat, lon)
                                           for lon, lat in poly.exterior.coords])
                            art = ax.fill(xs, ys, color=colour,
                                          alpha=0.4, zorder=3, visible=False)[0]
                            osm_artists.append(art)
                except Exception:
                    continue
        print(f"    {len(osm_artists)} OSM artists rendered.")
    else:
        print("    Warning: No GeoDataFrame — OSM feature layer will be empty.")

    # ── LSF scatter ───────────────────────────────────────────────────────────
    sc_lsf = ax.scatter(df["lsf_col"], df["lsf_row"],
                        c="#39d353", s=8, alpha=0.4, linewidths=0,
                        zorder=4, visible=False)

    # ── LKP marker ───────────────────────────────────────────────────────────
    ax.plot(center_col, center_row, marker="*", color="#f0c040",
            markersize=18, markeredgecolor="#0d1117", markeredgewidth=1.2,
            zorder=6)

    # ── Search radius rings ───────────────────────────────────────────────────
    mean_r_px   = df["distance_from_lkp_m"].mean()         / RESOLUTION_M
    median_r_px = df["distance_from_lkp_m"].median()       / RESOLUTION_M
    p90_r_px    = df["distance_from_lkp_m"].quantile(0.90) / RESOLUTION_M

    rings = []
    for r, color, ls_style in [
        (median_r_px, "#4fc3f7", "--"),
        (mean_r_px,   "#f48fb1", "-"),
        (p90_r_px,    "#ce93d8", ":"),
    ]:
        circle = plt.Circle((center_col, center_row), r,
                             fill=False, edgecolor=color, linewidth=1.4,
                             linestyle=ls_style, zorder=5, visible=True)
        ax.add_patch(circle)
        rings.append(circle)

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")

    # ── Colourbar ─────────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.965, 0.25, 0.012, 0.5])
    sm = plt.cm.ScalarMappable(cmap="hot", norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Normalised POC", color="#8b949e", fontsize=8, fontfamily="monospace")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e", fontsize=7)

    # ── Legend ────────────────────────────────────────────────────────────────
    osm_legend = [Patch(color=c, label=k, alpha=0.7)
                  for k, c in FEATURE_COLOURS.items()]
    standard_legend = [
        Line2D([0],[0], marker="*", color="w", markerfacecolor="#f0c040",
               markersize=12, label="LKP"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#39d353",
               markersize=6,  label="LSF survivors"),
        Line2D([0],[0], color="#4fc3f7", linewidth=1.4, linestyle="--", label="50th pct radius"),
        Line2D([0],[0], color="#f48fb1", linewidth=1.4, linestyle="-",  label="Mean radius"),
        Line2D([0],[0], color="#ce93d8", linewidth=1.4, linestyle=":",  label="90th pct radius"),
    ]
    ax.legend(handles=standard_legend + osm_legend, loc="lower right",
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#e6edf3", fontsize=7.5, framealpha=0.88,
              ncol=2)

    # ── Stats annotation ──────────────────────────────────────────────────────
    gravity_pct = df["gravity_bias"].mean() * 100
    stats_text  = (
        f"Subjects:  {NUM_SURVIVORS:,}\n"
        f"Mean R:    {df['distance_from_lkp_m'].mean():.0f} m\n"
        f"Median R:  {df['distance_from_lkp_m'].median():.0f} m\n"
        f"90th pct:  {df['distance_from_lkp_m'].quantile(0.90):.0f} m\n"
        f"Downhill:  {gravity_pct:.1f}%\n"
        f"Top driver: {df['top_logit_driver'].value_counts().idxmax()}\n"
        f"\nWeights (optimised):\n"
        f"  slope={w['slope']:.3f}  dist={w['dist']:.3f}\n"
        f"  feat={w['feat']:.3f}"
    )
    ax.text(0.01, 0.99, stats_text,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, fontfamily="monospace", color="#e6edf3",
            bbox=dict(facecolor="#161b22", edgecolor="#30363d",
                      alpha=0.85, boxstyle="round,pad=0.5"))

    # ── CheckButtons ──────────────────────────────────────────────────────────
    check_ax = fig.add_axes([0.01, 0.30, 0.155, 0.38], facecolor="#2d333b")
    check_ax.set_title("LAYERS", color="#8b949e", fontsize=9,
                        fontfamily="monospace", pad=6)

    labels_check = ["Elevation Map", "OSM Features", "POC Heatmap",
                    "LSF Points",    "Search Rings"]
    initial_vis  = [False, False, False, False, True]
    layer_map    = [im_elev, osm_artists, im_poc, sc_lsf, rings]

    check = CheckButtons(check_ax, labels_check, initial_vis)
    for text in check.labels:
        text.set_color("#e6edf3")
        text.set_fontsize(9)
        text.set_fontfamily("monospace")
    try:
        check.ax.set_facecolor("#2d333b")
        for span in check.ax.patches:
            span.set_facecolor("#3d444d")
            span.set_edgecolor("#8b949e")
    except Exception:
        pass

    def toggle(label):
        idx   = labels_check.index(label)
        layer = layer_map[idx]
        if isinstance(layer, list):
            new_vis = not layer[0].get_visible() if layer else True
            for item in layer:
                item.set_visible(new_vis)
        else:
            layer.set_visible(not layer.get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(toggle)

    # ── Info panel ────────────────────────────────────────────────────────────
    info_ax = fig.add_axes([0.01, 0.05, 0.155, 0.22], facecolor="#161b22")
    info_ax.axis("off")
    top_feat = df["specific_feature_label"].value_counts().head(3)
    info_lines = ["TOP FEATURES", "─" * 16]
    for feat, cnt in top_feat.items():
        info_lines.append(f"{feat:<10} {cnt/NUM_SURVIVORS*100:4.1f}%")
    info_lines += [
        "", "WEIGHT SPLIT", "─" * 16,
        f"Slope    {df['slope_pct'].mean()*100:4.1f}%",
        f"Distance {df['distance_pct'].mean()*100:4.1f}%",
        f"Feature  {df['feature_pct'].mean()*100:4.1f}%",
    ]
    info_ax.text(0.05, 0.97, "\n".join(info_lines),
                 va="top", ha="left", fontsize=8,
                 fontfamily="monospace", color="#8b949e",
                 transform=info_ax.transAxes)

    plt.show()

# =============================================================================
# 6. MAIN
# =============================================================================
def main(theta_hat=None):
    if theta_hat is None:
        print("    Warning: No optimised weights received — using baseline (0.5, 0.3, 0.2).")
        theta_hat = {"slope": 0.5, "dist": 0.3, "feat": 0.2}

    print("--- [Step 4] Loading environment data ---")
    elevation, heatmap, slope_p, dist_p, bounds, crs_str, gdf = load_all()

    print(f"--- Simulating {NUM_SURVIVORS:,} survivors with optimised weights ---")
    print(f"    slope={theta_hat['slope']:.4f}  dist={theta_hat['dist']:.4f}  "
          f"feat={theta_hat['feat']:.4f}")
    df, poc, lkp_coords, lkp_elev = simulate_survivors(
        elevation, heatmap, slope_p, dist_p, bounds, crs_str, theta_hat
    )

    print_explainability(df, lkp_elev, theta_hat)

    csv_cols = [
        "survivor_id", "lkp_latitude", "lkp_longitude",
        "lsf_latitude", "lsf_longitude",
        "distance_from_lkp_m", "lkp_elevation_m", "lsf_elevation_m",
        "elevation_delta_m", "gravity_bias",
        "slope_pct", "distance_pct", "feature_pct",
        "top_logit_driver", "specific_feature_label", "poc_value",
        "weight_slope", "weight_dist", "weight_feat",
    ]
    df[csv_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"--- CSV saved: {OUTPUT_CSV} ({len(df):,} rows) ---\n")

    print("--- Launching interactive GUI ---")
    launch_gui(df, elevation, heatmap, slope_p, dist_p, poc, lkp_coords,
               gdf, bounds, crs_str, theta_hat)

    return df

if __name__ == "__main__":
    main()
