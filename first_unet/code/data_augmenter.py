import torchio as tio
import numpy as np
import os
from load_and_clean_data import load_and_clean_data
from centring import centring
import random

def save_transformed_data(data, labels, transform, folder, site_name):
    os.makedirs(folder, exist_ok=True)
    transformed_data = []
    transformed_labels = []

    for i, (image, label) in enumerate(zip(data, labels)):
        subject = tio.Subject(x=tio.ScalarImage(tensor=image), y=tio.LabelMap(tensor=label))
        transformed_subject = transform(subject)
        
        # Extract the transformed numpy array from the subject
        transformed_image = transformed_subject['x']['data'].numpy()
        transformed_label = transformed_subject['y']['data'].numpy()
        
        transformed_data.append(transformed_image)
        transformed_labels.append(transformed_label)

        if i % 10 == 9:
            print(f"{site_name} | Done {i + 1} labels out of {len(data)}", flush=True)
        
    # Save all transformed images and labels as a single numpy array
    transformed_data_array = np.stack(transformed_data)
    transformed_labels_array = np.stack(transformed_labels)

    np.save(os.path.join(folder, f'x_train_{site_name}.npy'), transformed_data_array)
    np.save(os.path.join(folder, f'y_train_{site_name}.npy'), transformed_labels_array)
    print(f"Saved transformed images and labels for {site_name} to {folder}")

    # Saving one random transformed example
    check_folder = '/home/chri6020/first_unet/transformed_samples/'
    random_index = random.randint(0, len(data) - 1)
    random_subject = transform(tio.Subject(x=tio.ScalarImage(tensor=data[random_index]), y=tio.LabelMap(tensor=labels[random_index])))
    
    # Save the random transformed 'X' image
    x_random_path = os.path.join(check_folder, f'random_transformed_X_{site_name}.nii.gz')
    random_subject['x'].save(x_random_path)

    # Save the random transformed 'Y' label map
    y_random_path = os.path.join(check_folder, f'random_transformed_Y_{site_name}.nii.gz')
    random_subject['y'].save(y_random_path)

    print(f"Saved one random transformed example for {site_name} to {check_folder}")

# Define the transformation pipelines
simple_pipeline = tio.Compose([
    tio.RandomAffine(),
    tio.RandomElasticDeformation(),
])

mri_pipeline = tio.Compose([
    tio.RandomAffine(),
    tio.RandomElasticDeformation(),
    tio.RandomBiasField(),
])

# Paths for saving the transformed datasets

# Load the raw data
site_list = np.load("/home/chri6020/first_unet/metrics/sorted_sites.npy")

# Loop over sites and apply transformations
for site_name in site_list:
    print(f"{site_name} | Started data augmentation", flush=True)
    X_train = load_and_clean_data(site_name, 'X', 'train')
    y_train = load_and_clean_data(site_name, 'y', 'train')
    X_train = centring(X_train)
    y_train = np.where((y_train == 17) | (y_train == 53), 1, 0)
    print(f"{site_name} | Loaded data", flush=True)

    # Since the data is 3D and torchio works with 4D (channels, i, j, k), 
    # you need to add a channel dimension to the 3D images
    X_train = X_train[:, np.newaxis, ...]  # Convert from (N, i, j, k) to (N, C=1, i, j, k)
    y_train = y_train[:, np.newaxis, ...]

    # Paths for saving the transformed datasets
    save_path_pipeline1 = '/home/chri6020/subcortical/simple'
    save_path_pipeline2 = '/home/chri6020/subcortical/mri'

    # Apply the first pipeline and save the transformed data
    save_transformed_data(X_train, y_train, simple_pipeline, save_path_pipeline1, site_name)

    # Apply the second pipeline and save the transformed data
    save_transformed_data(X_train, y_train, mri_pipeline, save_path_pipeline2, site_name)

