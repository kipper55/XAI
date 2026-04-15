


# region_processing.py
import os
import numpy as np
import shapely.geometry
import rioxarray
import geopandas as gpd
import pandas as pd
from skimage.transform import resize
from sarenv import DataGenerator
from subject_behaviour import LPBExplainability

def export_final_labeled_ben_nevis():
    data_gen = DataGenerator()

    # 1. Paths
    tif_path   = r"C:\Users\antvi\PYTHON_PROJECTS\SAREnv\sarenv_dataset\ben_nevis\ben_nevis_geo.tif"
    output_dir = r"C:\Users\antvi\PYTHON_PROJECTS\SAREnv\sarenv_dataset\ben_nevis"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Open Master Terrain
    print("--- Opening COP30 Terrain ---")
    rds        = rioxarray.open_rasterio(tif_path)
    target_shape = (rds.rio.height, rds.rio.width)
    bounds     = rds.rio.bounds()
    elevation  = rds.values[0] if rds.values.ndim == 3 else rds.values
    ben_nevis_polygon = shapely.geometry.box(*bounds)

    # 3. Generate Environment (OSM Fetch)
    print("--- Initializing Environment & Fetching OSM Data ---")
    env = data_gen.generate_environment_from_polygon(polygon=ben_nevis_polygon, meter_per_bin=30)

    # 4. Labeled Feature Extraction
    print("--- Exporting Labeled features.geojson ---")
    all_layers = []
    for key, value in env.features.items():
        if value is None:
            continue

        layer_gdf = None

        if isinstance(value, gpd.GeoDataFrame):
            layer_gdf = value.copy()
            layer_gdf['feature_type'] = key
        elif isinstance(value, gpd.GeoSeries):
            layer_gdf = gpd.GeoDataFrame(geometry=value, crs=env.crs)
            layer_gdf['feature_type'] = key
        elif isinstance(value, list) or hasattr(value, 'geoms'):
            geoms = value.geoms if hasattr(value, 'geoms') else value
            harvested = []
            for g in geoms:
                if isinstance(g, shapely.geometry.base.BaseGeometry):
                    harvested.append({'geometry': g, 'feature_type': key})
            if harvested:
                layer_gdf = gpd.GeoDataFrame(harvested, crs="EPSG:32630")

        if layer_gdf is not None and not layer_gdf.empty:
            all_layers.append(layer_gdf)

    if all_layers:
        gdf_to_save = pd.concat(all_layers, ignore_index=True)
        if not isinstance(gdf_to_save, gpd.GeoDataFrame):
            gdf_to_save = gpd.GeoDataFrame(gdf_to_save, crs="EPSG:32630")
        gdf_to_save = gdf_to_save.to_crs("EPSG:4326")
        gdf_to_save.to_file(os.path.join(output_dir, "features.geojson"), driver='GeoJSON')
        print(f"SUCCESS: Saved {len(gdf_to_save)} labeled features.")
    else:
        print("Warning: No features found to label.")

    # 5. Terrain Processing — Raw Slope
    print("--- Processing Terrain Layers ---")
    dy, dx     = np.gradient(elevation, 30, 30)
    slope_deg  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    np.save(os.path.join(output_dir, "slope.npy"), slope_deg)

    # 6. Slope Probability (SPS penalty) — owned here, consumed by optimiser
    print("--- Computing Slope Probability ---")
    center_y, center_x = elevation.shape[0] // 2, elevation.shape[1] // 2
    lkp_elev   = elevation[center_y, center_x]
    logic      = LPBExplainability()
    slope_p    = logic.calculate_slope_penalty(elevation, lkp_elevation=lkp_elev)
    np.save(os.path.join(output_dir, "slope_p.npy"), slope_p)

    # 7. Distance Probability — owned here, consumed by optimiser
    print("--- Computing Distance Decay ---")
    y, x   = np.indices(elevation.shape)
    dist_p = np.exp(-np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / 300)
    dist_p /= dist_p.max()
    np.save(os.path.join(output_dir, "dist_p.npy"), dist_p)

    # 8. Combined Heatmap
    print("--- Creating Aligned Heatmap ---")
    lib_heatmap          = env.get_combined_heatmap()
    lib_heatmap_resized  = resize(lib_heatmap, target_shape, anti_aliasing=True, mode='reflect')
    terrain_penalty      = np.exp(-0.15 * slope_deg)
    super_heatmap        = lib_heatmap_resized * terrain_penalty
    super_heatmap       /= np.sum(super_heatmap)
    np.save(os.path.join(output_dir, "heatmap.npy"), super_heatmap)

    print(f"--- Complete: All files saved to {output_dir} ---")
    return True

if __name__ == "__main__":
    export_final_labeled_ben_nevis()