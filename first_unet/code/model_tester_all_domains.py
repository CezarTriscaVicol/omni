import torch
from unet3d import UNet3D
import numpy as np
import matplotlib.pyplot as plt
from numpy_dataset import numpy_dataset
from diceLoss import DiceLoss
from torch.utils.data import DataLoader
import seaborn as sns
import pandas as pd
from dice_coefficient import dice_coefficient
from centring import centring
import os

def print_graphs(model_names, aggregate_model_name, compute=True):
    
    site_list = np.load('/home/chri6020/first_unet/metrics/sorted_sites.npy')
    #site_list = ["Caltech","NYU","SDSU","Yale","CMU","OHSU","Stanford","KKI","Olin","Trinity","Leuven","Pitt","UCLA","MaxMun","SBL","UM"] 
    #site_list = ["Caltech","NYU","SDSU","Yale"]
    
    if compute:
        for model_name in model_names:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
            model = UNet3D(in_channels=1, init_features=4, out_channels=1).to(device)

            epoch_number_to_load = 50

            model_path = f'/home/chri6020/first_unet/saved_models/{model_name}/model_epoch_{epoch_number_to_load}.pth'
            model.load_state_dict(torch.load(model_path))

            dice_loss_func = DiceLoss()
            site_dices = [] 
            site_dice_scores = {}

            def test(net, test_dataloader, loss_func):
                net.eval()  # Put the model in eval mode
                total_loss = 0
                dice_vals = []  # List to store all dice coefficients for variance calculation
                with torch.no_grad():  # So no gradients accumulate
                    for batch_idx, (data, target) in enumerate(test_dataloader):
                        data, target = data.to(device), target.to(device)
                        outputs = net(data)  # Forward pass
                        outputs = torch.squeeze(outputs, 1)
                        target = target.float()
                        loss = loss_func(outputs, target)  # Loss calculation
                        total_loss += loss.item()
                        dice_val = dice_coefficient(outputs, target)
                        dice_vals.append(dice_val.item())  # Store dice coefficient

                av_loss = total_loss / len(test_dataloader)
                av_dice = sum(dice_vals) / len(dice_vals)
                return av_loss, av_dice, dice_vals  # Return the list of dice coefficients

            for site in site_list:
                print(f"Started site {site}", flush=True)

                # Load test data
                X_test = np.load(f'/home/chri6020/subcortical/X_{site}_test.npy')
                y_test = np.load(f'/home/chri6020/subcortical/y_{site}_test.npy')
                X_test = centring(X_test)
                y_test = np.where((y_test == 17) | (y_test == 53), 1, 0)

                print("Shape of X_test and y_test: ", X_test.shape, " | ", y_test.shape)

                #if site != "NYU":
                if False:
                    # Load training data as well for non-NYU sites
                    X_train = np.load(f'/home/chri6020/subcortical/X_{site}_train.npy')
                    y_train = np.load(f'/home/chri6020/subcortical/y_{site}_train.npy')
                    X_train = centring(X_train)
                    y_train = np.where((y_train == 17) | (y_train == 53), 1, 0)

                    # Concatenate train and test data
                    X_combined = np.concatenate((X_train, X_test), axis=0)
                    y_combined = np.concatenate((y_train, y_test), axis=0)
                else:
                    # For NYU, only use test data
                    X_combined = X_test
                    y_combined = y_test

                label_mapping = {0: 0, 1: 1}
                combined_dataset = numpy_dataset(X_combined, y_combined, label_mapping)
                combined_dataloader = DataLoader(combined_dataset, batch_size=3, shuffle=False, drop_last=True)

                print("Started computing dice coefficient.", flush = True)
                av_loss, av_dice, dice_list = test(model, combined_dataloader, dice_loss_func)
                site_dices.append(av_dice)
                site_dice_scores[site] = dice_list  # Store the list of Dice coefficients

            os.makedirs(f'/home/chri6020/first_unet/comparative_results/{model_name}', exist_ok=True)
            # After the loop, instead of saving averages and stds, we save the dictionary of dice scores
            np.save(f'/home/chri6020/first_unet/comparative_results/{model_name}/site_dice_scores.npy', site_dice_scores)
    
    # Now, outside the compute block, we load and bundle the dice scores for all sites and models
    bundled_site_dice_scores = {site: [] for site in site_list}

    for model_name in model_names:
        # Load the saved dice scores for each model
        model_dice_scores = np.load(f'/home/chri6020/first_unet/comparative_results/{model_name}/site_dice_scores.npy', allow_pickle=True).item()
        for site, dice_list in model_dice_scores.items():
            bundled_site_dice_scores[site].extend(dice_list)

    all_dice_data = []
    for site, dice_list in bundled_site_dice_scores.items():
        for dice in dice_list:
            all_dice_data.append((site, dice))

    df_all_dices = pd.DataFrame(all_dice_data, columns=['Site', 'Dice Coefficient'])

    # Now you can plot using seaborn's boxplot or violinplot without specifying 'hue' for different models
    plt.figure(figsize=(15, 7))
    sns.boxplot(x='Site', y='Dice Coefficient', data=df_all_dices)
    plt.xlabel('Sites', fontsize=14)
    plt.ylabel('Dice Coefficient', fontsize=14)
    plt.title('Aggregated Dice Coefficient Distribution by Site', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'/home/chri6020/first_unet/saved_figures/{aggregate_model_name}_aggregated_boxPlot_DiceCoeff.png')
    plt.close()

    # Plotting and saving figures for each site, bundling data from all models
    for site in site_list:
        site_dice_scores = np.load(f'/home/chri6020/first_unet/comparative_results/{model_name}/site_dice_scores.npy', allow_pickle=True).item()
        all_dice_data = [(site, dice) for site, dice_list in site_dice_scores.items() for dice in dice_list]
        df_all_dices = pd.DataFrame(all_dice_data, columns=['Site', 'Dice Coefficient'])

        plt.figure(figsize=(15, 7))
        sns.boxplot(x='Site', y='Dice Coefficient', data=df_all_dices)

        plt.xlabel('Sites', fontsize=14)
        plt.ylabel('Dice Coefficient', fontsize=14)
        plt.title('Dice Coefficient Distribution by Site', fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'/home/chri6020/first_unet/saved_figures/{model_name}_boxPlot_DiceCoeff.png')
        plt.close()

print_graphs(['fed_avg1_E5', 'fed_avg1_Econv', 'fed_avg2_E5_4sites'], 'fed_avg', True)