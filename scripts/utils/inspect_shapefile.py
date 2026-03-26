import geopandas as gpd

# Load the shapefile
shp_path = '/Users/wen11/Documents/OceanEdge_Group/Marine_dataset/MARIDA/shapefiles/S2_1-12-19_48MYU.shp'
gdf = gpd.read_file(shp_path)

print("Shapefile CRS:", gdf.crs)
print("Number of features:", len(gdf))
print("Columns:", list(gdf.columns))
print("First few rows:")
print(gdf.head())
print("Geometry type:", gdf.geometry.type.iloc[0] if len(gdf) > 0 else "No features")
