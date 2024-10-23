import os
import importlib
import pandas as pd
import numpy as np
import ast
import tqdm
import sys
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from utils.bev_utils import plot_boxes, get_intersection_image_from_vehicleids

"""
This script is used to recreate the BEV images from the dataset data using for example a new resolution.
"""


def recreate_images(path: str, new_resolution: int, radius: int = 50):
    # open the dataset
    dataset = pd.read_csv(os.path.join(path, 'dataset.csv')).reset_index(drop=True)
    dataset['detected_vehicle_infos'] = dataset['detected_vehicle_infos'].apply(ast.literal_eval)
    dataset['intersection'] = dataset['intersection'].apply(ast.literal_eval)
    os.mkdir(os.path.join(path, 'detected_bev_images'))
    for idx, row in tqdm.tqdm(dataset.iterrows(), total=len(dataset)):
        intersection_name, values = list(dataset['intersection'].iloc[0].items())[0]
        intersection_dict = {intersection_name: values}

        vehicles_dict = row['detected_vehicle_infos']

        # plot the BEV image
        fig = get_intersection_image_from_vehicleids(intersection_name, intersection_dict, vehicles_dict, radius, target_size=new_resolution)
        plt.draw()
        fig.savefig(os.path.join(path, 'detected_bev_images', f'{row["id"]}.png'), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
    dataset['complete_vehicle_infos'] = dataset['complete_vehicle_infos'].apply(ast.literal_eval)
    os.mkdir(os.path.join(path, 'complete_bev_images')) 
    for idx, row in tqdm.tqdm(dataset.iterrows(), total=len(dataset)):
        intersection_name, values = list(dataset['intersection'].iloc[0].items())[0]
        intersection_dict = {intersection_name: values}

        vehicles_dict = row['complete_vehicle_infos']

        # plot the BEV image
        fig = get_intersection_image_from_vehicleids(intersection_name, intersection_dict, vehicles_dict, radius, target_size=new_resolution)
        plt.draw()
        fig.savefig(os.path.join(path, 'complete_bev_images', f'{row["id"]}.png'), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
    # get the dataset keys that contain 'fco_time_' and iterate over them
    for key in dataset.keys():
        if 'fco_time_' in key:
            os.mkdir(os.path.join(path, key))
            for idx, row in tqdm.tqdm(dataset.iterrows(), total=len(dataset)):
                intersection_name, values = list(dataset['intersection'].iloc[0].items())[0]
                intersection_dict = {intersection_name: values}
                
                # check if current row[key] is nan
                if pd.isna(row[key]):
                    continue 
                else:
                    vehicles_dict = ast.literal_eval(row[key])

                # plot the BEV image
                fig = get_intersection_image_from_vehicleids(intersection_name, intersection_dict, vehicles_dict, radius, target_size=new_resolution)
                plt.draw()
                fig.savefig(os.path.join(path, key, f'{row["id"]}.png'), bbox_inches='tight', pad_inches=0)
                plt.close(fig)



if __name__ == "__main__":
    path = 'data/i3040_newdetector_10p_512_val'
    new_resolution = 512
    radius = 100

    # check if the path has a the folders already deleted
    if os.path.exists(os.path.join(path, 'complete_bev_images')):
        raise Exception('The folder complete_bev_images already exists. Delete it first to recreate the images.')

    if os.path.exists(os.path.join(path, 'detected_bev_images')):
        raise Exception('The folder complete_bev_images already exists. Delete it first to recreate the images.')
    
    # create a .txt file called changes.txt in the path with the content 'changed resolution to {new_resolution}'
    with open(os.path.join(path, 'changes.txt'), 'w') as f:
        f.write(f'changed resolution to {new_resolution}')
    
    recreate_images(path, new_resolution, radius)

    # get the