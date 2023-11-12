import matplotlib.pyplot as plt
import os
import numpy as np

def collate_and_plot(model_name, site_list, max_comm_rounds=-1):

    directory = f'/home/chri6020/first_unet/graphs/{model_name}'
    os.makedirs(directory, exist_ok=True)
    
    for metric in ['train_loss', 'test_loss', 'train_dice', 'test_dice']:
        plt.figure(figsize=(10, 6))
        
        all_site_data = {site: [] for site in site_list}
        max_length_per_round = {}  # Store the max length of data for each round
        
        # First, find the max length of data in each round
        round_idx = 0
        while True:
            max_length = 0
            round_data_exists = False
            for site in site_list:
                round_directory = f'/home/chri6020/first_unet/metrics/{model_name}/{site}/round_{round_idx}'
                file_path = f'{round_directory}/{metric}.npy'
                if os.path.exists(file_path):
                    round_data_exists = True
                    data_length = len(np.load(file_path).flatten())
                    max_length = max(max_length, data_length)
            
            if not round_data_exists:
                break  # Exit if no data exists for any site in the current round
            
            max_length_per_round[round_idx] = max_length
            round_idx += 1

            if max_comm_rounds != -1 and round_idx >= max_comm_rounds:
                break
        
        # Now, load, pad and concatenate the data
        for round_idx, max_length in max_length_per_round.items():
            for site in site_list:
                round_directory = f'/home/chri6020/first_unet/metrics/{model_name}/{site}/round_{round_idx}'
                file_path = f'{round_directory}/{metric}.npy'
                if os.path.exists(file_path):
                    data = np.load(file_path).flatten()
                    padded_data = np.pad(data, (0, max_length - len(data)), 'constant', constant_values=np.nan)
                    all_site_data[site].extend(padded_data)
                else:
                    # If file doesn't exist, add nans for the current round
                    all_site_data[site].extend([np.nan] * max_length)
        
        # Now plot the data
        for site, data in all_site_data.items():
            plt.plot(data, label=f'Site {site}', alpha=0.5)
        
        # Plot average epoch metrics
        avg_data = np.nanmean(np.array(list(all_site_data.values())), axis=0)
        plt.plot(avg_data, label='Average', linewidth=2, color='black')
        
        plt.title(f'{metric.replace("_", " ").title()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f'{directory}/{metric}_epochs.png')
        plt.close()

    for metric in ['comm_round_test_loss', 'comm_round_test_dice']:
        plt.figure(figsize=(10, 6))
        
        # Plot communication round metrics
        for site in site_list:
            data = np.load(f'metrics/{model_name}/{site}/{metric}.npy')
            plt.plot(data, label=f'Site {site}', alpha=0.5)
        
        # Plot average communication round metrics
        all_data = [np.load(f'metrics/{model_name}/{site}/{metric}.npy') for site in site_list]
        avg_data = np.mean(np.array(all_data), axis=0)
        plt.plot(avg_data, label='Average', linewidth=2, color='black')
        
        plt.title(f'{metric.replace("_", " ").title()} over Communication Rounds')
        plt.xlabel('Communication Round')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f'{directory}/{metric}_comm_rounds.png')
        plt.close()