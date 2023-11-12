import pandas as pd
import numpy as np
import os
import imgkit
import matplotlib.pyplot as plt
from pandas.plotting import table

def save_df_as_image(df, filename):
    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(df.shape[1], df.shape[0]))  # Adjust figure size as needed

    # Hide the axes
    ax.axis('off')

    # Create the table and position it in the upper-left corner of the figure
    tbl = table(ax, df, loc='upper left')

    # Adjust the table's font size, scale, and cell padding as needed
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.2)  # May need to adjust scaling to fit your figure
    for key, cell in tbl.get_celld().items():
        cell.set_linewidth(0)  # Hide the cell borders

    # Save the figure
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)  # Adjust padding as needed
    plt.close(fig)

def print_dice_stats_to_csv_and_image(model_names, output_folder):
    print("Started printing", flush=True)
    output_csv_path = os.path.join(output_folder, 'dice_stats.csv')
    output_image_path = os.path.join(output_folder, 'dice_stats.png')
    site_list = np.load(os.path.join(output_folder, 'sorted_sites.npy'))
    stats_data = []

    for model_name in model_names:
        model_path = f'/home/chri6020/first_unet/comparative_results/{model_name}/site_dice_scores.npy'
        if os.path.exists(model_path):
            site_dice_scores = np.load(model_path, allow_pickle=True).item()
            for site in site_list:
                if site in site_dice_scores:
                    # Calculate mean and variance
                    dice_scores = site_dice_scores[site]
                    mean_dice = np.mean(dice_scores)
                    var_dice = np.var(dice_scores)
                    stats_data.append((model_name, site, mean_dice, var_dice))
    print(stats_data)
    # Convert to DataFrame
    df_stats = pd.DataFrame(stats_data, columns=['Model', 'Site', 'Mean Dice Coefficient', 'Variance Dice Coefficient'])

    # Pivot the DataFrame to get the desired format
    df_pivot = df_stats.pivot(index='Model', columns='Site', values=['Mean Dice Coefficient', 'Variance Dice Coefficient'])

    # Flatten MultiIndex columns and concatenate the level values with an underscore
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

    # Save to CSV
    df_pivot.to_csv(output_csv_path)

# Define the output folder
output_folder = '/home/chri6020/first_unet/metrics'

# Call the function
print_dice_stats_to_csv_and_image(['Dice1', 'Dice2', 'Dice3', 'Dice4', 'Dice5', 'fed_avg1_E5', 'fed_avg1_Econv'], output_folder)
