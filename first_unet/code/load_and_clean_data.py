import numpy as np
def remove_badlabels(images, site):
    bad_indices = {
        'KKI': [1],
        'NYU': [30],
        'UCLA': [11],
        'UM': [1, 3, 10, 19, 24]
    }
    
    if site in bad_indices:
        mask = np.ones(len(images), dtype=bool)  
        mask[bad_indices[site]] = False  
        cleaned_images = images[mask]  
        return cleaned_images
    else:
        return images
    
def load_and_clean_data(site_name, data_type, file_type, augmentation_type='none'):
    file_path = f'/home/chri6020/subcortical/{augmentation_type}/{data_type}_{site_name}_{file_type}.npy'
    data = np.load(file_path)
    if file_type == 'test':
        data = remove_badlabels(data, site_name)
    return data
