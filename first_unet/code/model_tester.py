import torch
from unet3d import UNet3D
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = UNet3D(in_channels=1, init_features=4, out_channels=1).to(device)

epoch_number_to_load = 50

model_path = f'/home/chri6020/first_unet/saved_models/model_Dice_epoch_{epoch_number_to_load}.pth'
model.load_state_dict(torch.load(model_path))

X_train = np.load(f'/home/chri6020/subcortical/X_NYU_train.npy')
X_test = np.load(f'/home/chri6020/subcortical/X_NYU_test.npy')
y_train = np.load(f'/home/chri6020/subcortical/y_NYU_train.npy')
y_test = np.load(f'/home/chri6020/subcortical/y_NYU_test.npy')

y_train = np.where((y_train == 17) | (y_train == 53), 1, 0)
y_test = np.where((y_test == 17) | (y_test == 53), 1, 0)

print("Shape of X_train: ", X_train.shape, flush=True)
print("Shape of X_test:  ", X_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_test:  ", y_test.shape)

indices = range(0, 128, 3)

num_columns = 9
num_rows = int(np.ceil(len(indices) / (num_columns / 3)))

input_tensor = torch.tensor(X_train[0][None, None, ...], dtype=torch.float32).to(device)

with torch.no_grad(): 
    output = model(input_tensor)
    # Convert y_train to tensor

y_true_tensor = torch.tensor(y_train[0][None, None, ...], dtype=torch.float32).to(device)

def dice_coefficient(pred, target):
    """Calculate Dice coefficient."""
    smooth = 1e-7  # To avoid division by zero
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
# Compute Dice loss
dice_coeff = dice_coefficient(y_true_tensor, output)

print(f"Dice coeff: {dice_coeff.item()}", flush=True)


predicted = output.squeeze().cpu().numpy()

print("Computed predicted, now printing", flush=True)

plt.figure(figsize=(15, num_rows * 3.5))  

for i, idx in enumerate(indices):
    # Display X slice
    plt.subplot(num_rows, num_columns, 3*i + 1)
    plt.imshow(X_train[0][idx], cmap='gray')
    plt.title(f'X Slice {idx}')

    # Display y slice (ground truth)
    plt.subplot(num_rows, num_columns, 3*i + 2)
    plt.imshow(y_train[0][idx], cmap='gray')
    plt.title(f'y Slice {idx}')

    # Display predicted slice
    plt.subplot(num_rows, num_columns, 3*i + 3)
    plt.imshow(predicted[idx], cmap='gray')
    plt.title(f'Predicted Slice {idx}')

plt.tight_layout()
fig_path = '/home/chri6020/first_unet/saved_figures/PredictedBrainSlices.png'
plt.savefig(fig_path)
plt.show()
