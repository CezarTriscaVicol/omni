import numpy as np
def centring(X):
    epsilon = 1e-7 
    X_mean = np.mean(X)
    X_std = np.std(X)
    X_standardized = (X - X_mean) / (X_std + epsilon)
    return X_standardized
