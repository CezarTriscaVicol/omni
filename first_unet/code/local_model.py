import torch
from torch.utils.data import DataLoader
from diceLoss import DiceLoss
from dice_coefficient import dice_coefficient
from numpy_dataset import numpy_dataset
import numpy as np
import time
from unet3d import UNet3D
import matplotlib.pyplot as plt
import os
from centring import centring
from load_and_clean_data import load_and_clean_data
import torchio as tio

# Define your augmentation pipelines
no_augmentation_pipeline = tio.Compose([])

simple_augmentation_pipeline = tio.Compose([])
# This will load data that already has affine and elastic transformations applied to it

mri_specific_augmentation_pipeline = tio.Compose([
    tio.RandomGhosting(p=0.5),  # Adds random ghosting artifacts
    tio.RandomSpike(p=0.5),  # Adds random spike artifacts
    tio.RandomBlur(p=0.5),  # Adds random blurring
    tio.RandomNoise(p=0.5),  # Adds random noise
    tio.RandomGamma(p=0.5),  # Adds random gamma transformations
])
# This will load data that already has affine, elastic and bias field transformations applied

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.count = 0
        

    def check(self, loss):
        if (self.best_loss - loss) > self.min_delta:
            self.best_loss = loss
            self.count = 0  # Reset count when we see improvement
        else:
            self.count += 1
        return self.count >= self.patience

class ModelTrainer:
    def __init__(self, model_name, site_name, augmentation_policy='none'):
        self.train_loss = []
        self.test_loss = []
        self.train_dice = []
        self.test_dice = []
        self.comm_round_test_loss = []
        self.comm_round_test_dice = [] 
        self.model_name = model_name
        self.site_name = site_name
        self.augmentation_policy = augmentation_policy
        self.augmentation_pipelines = {
            'none': no_augmentation_pipeline,
            'simple': simple_augmentation_pipeline,
            'mri': mri_specific_augmentation_pipeline
        }
    
    def save_transformed_samples(self, dataset, num_samples, folder):
        os.makedirs(folder, exist_ok=True)  # Ensure the directory exists
        for i, subject in enumerate(dataset):
            if i >= num_samples:  # Save only the specified number of samples
                break
            # Save the transformed 'X' image
            x_path = os.path.join(folder, f'transformed_X_{i}.nii.gz')
            tio.ScalarImage(tensor=subject['x']['data']).save(x_path)

            # Save the transformed 'Y' label map
            y_path = os.path.join(folder, f'transformed_Y_{i}.nii.gz')
            tio.LabelMap(tensor=subject['y']['data']).save(y_path)
            print(f"Saved transformed sample {i} to {folder}")

    def train_local_model(self, label_mapping, model, batch_size, steps):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        X_train = load_and_clean_data(self.site_name, 'X', 'train', augmentation_type=self.augmentation_policy)
        y_train = load_and_clean_data(self.site_name, 'y', 'train', augmentation_type=self.augmentation_policy)

        num_images = X_train.shape[0]

        X_test = load_and_clean_data(self.site_name, 'X', 'test')
        y_test = load_and_clean_data(self.site_name, 'y', 'test')
        
        X_train = centring(X_train)
        X_test = centring(X_test)

        y_train = np.where((y_train == 17) | (y_train == 53), 1, 0)
        y_test  = np.where((y_test == 17) | (y_test == 53), 1, 0)
        print(f"{self.model_name} | {self.site_name} | Loaded data", flush=True)

        # Define subject for torchio dataset, which includes both X and y with the same transformations
        subjects = []
        for i in range(len(X_train)):
            # Add a channel dimension to the 3D image tensor
            x_tensor = torch.tensor(X_train[i].astype(np.float32)).unsqueeze(0)  # Now x_tensor is 4D
            y_tensor = torch.tensor(y_train[i].astype(np.float32)).unsqueeze(0)  # Now y_tensor is 4D
            subject = tio.Subject(
                x=tio.ScalarImage(tensor=x_tensor),
                y=tio.LabelMap(tensor=y_tensor)
            )
            subjects.append(subject)

        # Define the torchio dataset with transformations
        training_set = tio.SubjectsDataset(subjects, transform=self.augmentation_pipelines[self.augmentation_policy])
        
        #self.save_transformed_samples(training_set, num_samples=5, folder='/home/chri6020/first_unet/transformed_samples/')

        # Use a custom collate function to properly batch Subjects
        def collate_fn(batch):
            xs = torch.stack([item['x']['data'] for item in batch], dim=0)  # Adds the batch dimension
            ys = torch.stack([item['y']['data'] for item in batch], dim=0)
            return xs, ys

        # Then use this in your DataLoader
        train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn, num_workers=4)
        
        test_dataset = numpy_dataset(X_test, y_test, label_mapping)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        print(f"{self.model_name} | {self.site_name} | Created Data Loaders", flush=True)

        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_func = DiceLoss()

        early_stopping = EarlyStopping(patience=5, min_delta=0.001)

        model.eval()
        total_test_loss = 0
        total_test_dice = 0
        batches = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_dataloader):
                data, target = data.to(device), target.to(device)
                batches += 1
                outputs = model(data)
                outputs = torch.squeeze(outputs, 1)
                target = target.float()
                loss = loss_func(outputs, target)
                total_test_loss += loss.item()
                
                dice_val = dice_coefficient(outputs, target)
                total_test_dice += dice_val.item()

        # Store the computed test values for this communication round
        self.comm_round_test_loss.append(total_test_loss / batches)
        self.comm_round_test_dice.append(total_test_dice / batches)
        # Lists to store values for this communication round
        round_train_loss = []
        round_test_loss = []
        round_train_dice = []
        round_test_dice = []

        step = 0
        while (steps == -1) or (step < steps):
            step += 1
            model.train()
            total_loss = 0
            total_dice = 0
            batches = 0
            for batch_idx, (data, target) in enumerate(train_dataloader):
                data, target = data.to(device), target.to(device)
                batches += 1
                optim.zero_grad()
                outputs = model(data)
                #outputs = torch.squeeze(outputs, 1)
                target = target.float()
                #print(f"train {data.shape} | {outputs.shape} | {target.shape}")
                loss = loss_func(outputs, target)
                loss.backward()
                optim.step()
                total_loss += loss.item()

                dice_val = dice_coefficient(outputs, target)
                total_dice += dice_val.item()

            av_loss = total_loss / batches
            av_dice = total_dice / batches
            round_train_loss.append(av_loss)
            round_train_dice.append(av_dice)
            print(f'{self.model_name} | {self.site_name} | Step {step} | Training set: Average loss: {av_loss:.4f}, Average Dice Coeff: {av_dice:.4f}', flush=True)

            model.eval()
            total_test_loss = 0
            total_test_dice = 0
            batches = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_dataloader):
                    data, target = data.to(device), target.to(device)
                    batches += 1
                    outputs = model(data)
                    outputs = torch.squeeze(outputs, 1)
                    target = target.float()
                    #print(f"test {data.shape} | {outputs.shape} | {target.shape}")
                    loss = loss_func(outputs, target)
                    total_test_loss += loss.item()
                    dice_val = dice_coefficient(outputs, target)
                    total_test_dice += dice_val.item()

            av_test_loss = total_test_loss / batches
            av_test_dice = total_test_dice / batches
            round_test_loss.append(av_test_loss)
            round_test_dice.append(av_test_dice)
            print(f'{self.model_name} | {self.site_name} | Step {step} | Testing set: Average loss: {av_test_loss:.4f}, Average Dice Coeff: {av_test_dice:.4f}', flush=True)

            if (steps == -1) and early_stopping.check(av_test_loss):
                print(f"{self.model_name} | {self.site_name} | Early stopping at step: {step}")
                break

        self.train_loss.append(round_train_loss)
        self.test_loss.append(round_test_loss)
        self.train_dice.append(round_train_dice)
        self.test_dice.append(round_test_dice)

        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model, num_images
    
    def get_metrics(self):
        return self.train_loss, self.test_loss, self.train_dice, self.test_dice, self.comm_round_test_loss, self.comm_round_test_dice
    
    def save_metrics(self):
        # Ensure the directory exists
        directory = f'/home/chri6020/first_unet/metrics/{self.model_name}/{self.site_name}'
        os.makedirs(directory, exist_ok=True)

        # Save each communication round's data as a separate file
        for i, (train_loss, test_loss, train_dice, test_dice) in enumerate(zip(self.train_loss, self.test_loss, self.train_dice, self.test_dice)):
            round_directory = f'{directory}/round_{i}'
            os.makedirs(round_directory, exist_ok=True)
            np.save(f'{round_directory}/train_loss.npy', np.array(train_loss))
            np.save(f'{round_directory}/test_loss.npy', np.array(test_loss))
            np.save(f'{round_directory}/train_dice.npy', np.array(train_dice))
            np.save(f'{round_directory}/test_dice.npy', np.array(test_dice))

        # Save communication round test metrics
        np.save(f'{directory}/comm_round_test_loss.npy', np.array(self.comm_round_test_loss))
        np.save(f'{directory}/comm_round_test_dice.npy', np.array(self.comm_round_test_dice))

