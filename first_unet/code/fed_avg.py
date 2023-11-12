import torch
import numpy as np
from unet3d import UNet3D 
from local_model import ModelTrainer  
from collate_and_plot import collate_and_plot
import os

def update_model_with_weighted_average(model, models, weights):
    with torch.no_grad():  
        sum_weighted_params = [torch.zeros_like(param) for param in models[0].parameters()]
        total_weight = sum(weights) 
        for model_site, weight in zip(models, weights):
            for i, param in enumerate(model_site.parameters()):
                sum_weighted_params[i] += param.data * weight 
        
        for i, param in enumerate(model.parameters()):
            param.data.copy_(sum_weighted_params[i] / total_weight) 

def communication_rounds(model_name, initial_model, rounds, batch_size, steps, augmentation_policy):
    model = initial_model
    #site_list = ["Caltech","NYU","SDSU","Yale","CMU","OHSU","Stanford","KKI","Olin","Trinity","Leuven","Pitt","UCLA","MaxMun","SBL","UM"]
    #site_list = ["NYU", "UM", "UCLA", "Yale"]
    site_list = ["NYU"]
    trainers = {site: ModelTrainer(model_name, site, augmentation_policy) for site in site_list}

    for round_num in range(1, rounds + 1):
        models = []  
        weights = []  
        print(f"Communication Round {round_num}", flush=True)
        for site in site_list:
            label_mapping = {0: 0, 1: 1}
            model_site, num_images = trainers[site].train_local_model(label_mapping=label_mapping,
                                                               model=model, batch_size=batch_size, steps=steps)
            models.append(model_site)
            weights.append(num_images) # Choose this for weighted average (didnt work last time)
            #weights.append(1)
        update_model_with_weighted_average(model, models, weights)  # Update the model with the weighted average of the parameters
        for site in site_list:
            trainers[site].save_metrics()
        
        collate_and_plot(model_name, site_list)
        
        directory = f'/home/chri6020/first_unet/saved_models/{model_name}'
        os.makedirs(directory, exist_ok=True)
        model_path = f'{directory}/model_epoch_{round_num}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

    return model  

num_rounds = 50
batch_size = 8
steps = 5

for model_number in range(1, 6):
    initial_model = UNet3D(in_channels=1, init_features=4, out_channels=1)
    for augmentation_policy in ['none', 'simple', 'mri']:
        final_model = communication_rounds(f"fed_avg{model_number}_E{steps}_Aug{augmentation_policy}", initial_model, num_rounds, batch_size, steps, augmentation_policy)
                                                                                                                                                  # 'none' 'simple' 'mri'
    