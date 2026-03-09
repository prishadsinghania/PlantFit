import rasterio
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os
import glob

def process_tiff_zones(folder_path):
    print(f"--- 🛰️ SCIENTIFIC SATELLITE ZONE DELINEATION ---")
    
    search_path = os.path.join(folder_path, "*.tif*")
    tif_files = glob.glob(search_path)
    
    if not tif_files:
        raise FileNotFoundError(f"No TIFF files found in '{folder_path}'.")
        
    print(f"Found {len(tif_files)} satellite images. Extracting data...")
    
    all_data = []
    
    for file in tif_files:
        with rasterio.open(file) as dataset:
            ndvi = dataset.read(1)
            transform = dataset.transform
            
            # FAST VECTORIZED COORDINATE EXTRACTION
            # This generates the exact GPS coordinates for the entire image instantly
            cols, rows = np.meshgrid(np.arange(ndvi.shape[1]), np.arange(ndvi.shape[0]))
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            
            # Convert to flat numpy arrays
            lons = np.array(xs).flatten()
            lats = np.array(ys).flatten()
            ndvi_flat = ndvi.flatten()
            
            # Filter out clouds, water, and NaN values (> 0.05 is usually healthy dirt/plants)
            valid_mask = (ndvi_flat > 0.05) & (~np.isnan(ndvi_flat))
            
            valid_ndvi = ndvi_flat[valid_mask]
            valid_lons = lons[valid_mask]
            valid_lats = lats[valid_mask]
            
            # Append to our master list. 
            # We round to 5 decimal places (~1 meter precision) so pixels from different days snap together
            for lat, lon, val in zip(valid_lats, valid_lons, valid_ndvi):
                all_data.append([round(lat, 5), round(lon, 5), val])
                
    print("Calculating historical average NDVI across all 36 dates...")
    
    # Dump everything into a giant pandas DataFrame
    df_all = pd.DataFrame(all_data, columns=['lat', 'lon', 'ndvi'])
    
    # Group by the exact GPS coordinates and calculate the mean NDVI over the 5 years
    df_spatial = df_all.groupby(['lat', 'lon'])['ndvi'].mean().reset_index()
    df_spatial = df_spatial.rename(columns={'ndvi': 'avg_ndvi'})
    
    print(f"Consolidated into {len(df_spatial)} unique agricultural coordinates.")
    
    print("Running K-Means Clustering to identify High, Medium, and Low yielding zones...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_spatial['cluster'] = kmeans.fit_predict(df_spatial[['avg_ndvi']])
    
    # Sort the clusters so the highest actual NDVI average naturally becomes "High Yield"
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(cluster_centers)
    
    mapping = {
        sorted_idx[0]: 'Low Yield Zone', 
        sorted_idx[1]: 'Medium Yield Zone', 
        sorted_idx[2]: 'High Yield Zone'
    }
    df_spatial['zone'] = df_spatial['cluster'].map(mapping)
    
    # Clean up output
    output_df = df_spatial[['lat', 'lon', 'zone']]
    
    output_filename = "bondville_management_zones.csv"
    output_df.to_csv(output_filename, index=False)
    
    print(f"\n✅ SUCCESS! Scientific field delineation complete.")
    print(f"Zone map saved to '{output_filename}'.")
    
    print("\nZone Distribution summary:")
    print(output_df['zone'].value_counts())

if __name__ == "__main__":
    YOUR_FOLDER = "satellite_imgs" 
    process_tiff_zones(YOUR_FOLDER)