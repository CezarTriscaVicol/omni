import os
import numpy as np

# Folder path
folder_path = "/home/chri6020/subcortical/"

# Get all filenames in the folder
all_files = os.listdir(folder_path)

# Extract unique site names
site_names = set([name.split("_")[1] for name in all_files if '_' in name])

# Compute total size and number of images for each site
site_data = {}
for site in site_names:
    total_size = 0
    total_images = 0
    print(f"Starting site: {site}", flush=True)
    for preffix in ['X', 'y']:
        for suffix in ["train", "test", "train", "test"]:
            file_path = os.path.join(folder_path, f"{preffix}_{site}_{suffix}.npy")
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
                
                # Load the numpy file to get the shape
                data = np.load(file_path)
                total_images += data.shape[0]

    site_data[site] = (total_size, total_images)
    print(f"Done site: {site}, total images: {total_images}", flush=True)

# Rank sites by total size
sorted_sites = sorted(site_data.keys(), key=lambda x: site_data[x][0], reverse=True)
np.save('/home/chri6020/first_unet/metrics/sorted_sites.npy', sorted_sites)

# Print the results
for site in sorted_sites:
    print(f"Site: {site}, Total size: {site_data[site][0]} bytes, Number of images: {site_data[site][1]}")
