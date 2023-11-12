import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from unet3d import UNet3D
from numpy_dataset import numpy_dataset
from diceLoss import DiceLoss
from dice_coefficient import dice_coefficient
from centring import centring
import os

print("Started", flush=True)

X_train = np.load('/home/chri6020/subcortical/X_NYU_train.npy')
X_test = np.load('/home/chri6020/subcortical/X_NYU_test.npy')
y_train = np.load('/home/chri6020/subcortical/y_NYU_train.npy')
y_test = np.load('/home/chri6020/subcortical/y_NYU_test.npy')

print("Finished loading data", flush=True)

y_train = np.where((y_train == 17) | (y_train == 53), 1, 0)
y_test = np.where((y_test == 17) | (y_test == 53), 1, 0)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test:  ", X_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_test:  ", y_test.shape)

unique_values = np.unique(y_train) 
print("Unique y values: ", unique_values, flush=True)

indices = range(0, 128, 3)

num_columns = 6
num_rows = int(np.ceil(len(indices) / (num_columns / 2)))

plt.figure(figsize=(15, num_rows * 2.5))

for i, idx in enumerate(indices):
    plt.subplot(num_rows, num_columns, 2*i + 1)
    plt.imshow(X_train[0][idx], cmap='gray')
    plt.title(f'X Slice {idx}')

    plt.subplot(num_rows, num_columns, 2*i + 2)
    plt.imshow(y_train[0][idx], cmap='gray')
    plt.title(f'y Slice {idx}')

plt.tight_layout()
fig_path = '/home/chri6020/first_unet/saved_figures/BrainSlices.png'
plt.savefig(fig_path)
plt.show()

plt.figure(figsize=(10, 5))  

#print("Train Mean/Std before normalization: ", np.mean(X_train), np.std(X_train))
#print("Test Mean/Std before normalization: ", np.mean(X_test), np.std(X_test))

plt.subplot(1,2,1)
plt.hist(X_train.ravel())
plt.title('Before Normalisation')
plt.xlabel('Intensity Value')

X_train = centring(X_train)

plt.subplot(1,2,2)
plt.hist(X_train.ravel())
plt.title('After Normalisation')
plt.xlabel('Intensity Value')


fig_path = f'/home/chri6020/first_unet/saved_figures/Normalisation.png'
plt.savefig(fig_path)
plt.show()

X_test  = centring(X_test)

print("Finished Normalizing", flush=True)

label_mapping = {value: index for index, value in enumerate(unique_values)}
print("label_mapping: ", label_mapping)
num_classes = len(unique_values)

print("Finished processing targets", flush=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device, flush=True)

def train(net, dataloader, optim, loss_func, epoch):
    net.train()  # Put the network in train mode
    total_loss = 0
    total_dice = 0
    batches = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(dataloader):

        data, target = data.to(device), target.to(device)

        batches += 1
        optim.zero_grad()  # Zero the gradients
        outputs = net(data)  # Forward pass
        outputs = torch.squeeze(outputs, 1)
        target = target.float()
        loss = loss_func(outputs, target)  # Loss calculation
        loss.backward()  # Backward pass
        optim.step()  # Update weights
        total_loss += loss.item()

        dice_val = dice_coefficient(outputs, target)
        total_dice += dice_val.item()

        #if batch_idx % 100 == 0:  # Report stats every x batches
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(dataloader.dataset),
        #               100. * batch_idx / len(dataloader), loss.item()), flush=True)

    av_loss = total_loss / batches
    av_dice = total_dice / batches
    
    print('Training set: Average loss: {:.4f}, Average Dice Coeff: {:.4f}'.format(av_loss, av_dice), flush=True)
    global total_time_taken
    total_time_taken += time.time() - start_time
    return av_loss, av_dice

def test(net, test_dataloader, optim, loss_func, epoch):
    net.eval()  # Put the model in eval mode
    total_loss = 0
    total_dice = 0
    batches = 0
    with torch.no_grad():  # So no gradients accumulate
        for batch_idx, (data, target) in enumerate(test_dataloader):
            
            data, target = data.to(device), target.to(device)

            batches += 1
            outputs = net(data)  # Forward pass
            outputs = torch.squeeze(outputs, 1)
            target = target.float()
            loss = loss_func(outputs, target)  # Loss calculation
            total_loss += loss.item()
 
            dice_val = dice_coefficient(outputs, target)
            total_dice += dice_val.item()

        av_loss = total_loss / batches
        av_dice = total_dice / batches
        
        print('Test set: Average loss: {:.4f}, Average Dice Coeff: {:.4f}'.format(av_loss, av_dice))
        return av_loss, av_dice

def predict(net, test_dataloader):
    net.eval()  # Put the model in eval mode
    pred_store = []
    true_store = []
    with torch.no_grad():  # So no gradients accumulate
        for data, target in test_dataloader:
            outputs = net(data)  # Forward pass
            pred_store.extend(outputs.argmax(dim=1).tolist())
            true_store.extend(target.tolist())
    return pred_store, true_store

train_dataset = numpy_dataset(X_train, y_train, label_mapping)
test_dataset = numpy_dataset(X_test, y_test, label_mapping)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=True)

print("Created Data Loaders", flush = True)

for model_number in range(1, 6):
    
    model_name = f'Dice{model_number}'

    net = UNet3D(in_channels=1, init_features=4, out_channels=1).to(device)
    print(f"{model_name} | Created Unet", flush=True)

    # Calculate the number of trainable params
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{model_name} | Trainable params: ', params, flush=True)

    #number_of_positive_samples = (y_train == 1).sum()
    #number_of_negative_samples = (y_train == 0).sum()

    #pos_weight = number_of_negative_samples / number_of_positive_samples
    #print("Positive weight for BCE loss: ", pos_weight, flush=True)
    #pos_weight = torch.tensor([pos_weight], device=device)

    #class_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    class_loss = DiceLoss()
    optim = torch.optim.Adam(net.parameters(), lr=0.001)

    losses = []
    max_epochs = 100
    dice_scores = []
    total_time_taken = 0
    for epoch in range(1, max_epochs+1):
        print("Epoch: " + str(epoch))
        train_loss, train_dice = train(net, train_dataloader, optim, class_loss, epoch)
        test_loss, test_dice = test(net, test_dataloader, optim, class_loss, epoch)
        losses.append([train_loss, test_loss])
        dice_scores.append([train_dice, test_dice])
        # Save the model every 5 epochs
        if epoch % 2 == 0:
            directory = f'/home/chri6020/first_unet/saved_models/{model_name}'
            os.makedirs(directory, exist_ok=True)

            model_path = f'{directory}/model_epoch_{epoch}.pth'
            torch.save(net.state_dict(), model_path)
            print(f'Model saved to {model_path}')
        
    print(f"{model_name} | Average time taken per epoch: " + str(total_time_taken / max_epochs), flush=True)

    losses = np.array(losses).T
    print(losses.shape)
    its = np.linspace(1, max_epochs, max_epochs)

    dice_scores = np.array(dice_scores).T

    # Plotting
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(its, losses[0, :])
    plt.plot(its, losses[1, :])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'])

    plt.subplot(1, 2, 2)
    plt.plot(its, dice_scores[0, :])
    plt.plot(its, dice_scores[1, :])
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend(['Train', 'Test'])


    plt.tight_layout()
    directory = f'/home/chri6020/first_unet/graphs/{model_name}'
    os.makedirs(directory, exist_ok=True)
    fig_path = f'{directory}/LossDiceCoeffPlot.png'
    plt.savefig(fig_path)
    print(f'{model_name} | Figure saved to {fig_path}', flush=True)

    plt.show()

    net.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()