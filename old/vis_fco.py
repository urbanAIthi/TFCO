import os
import ast
import importlib
from typing import List
import matplotlib.patches as patches

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from create_fco_target import create_box
import cv2



def generate_fco_visualization(dataset_path: str):
    """
    This function will generate BEV images out of the FCO datasets.
    Different than the input images given into the network, these images have hte goal to
    visualize the fco, detected vehicles and undetected vehicles. in different colors.
    """
    # generate folder for the images
    os.makedirs(os.path.join(dataset_path, 'fco_visualization'), exist_ok=True)

    df = pd.read_csv(os.path.join(dataset_path, 'dataset.csv'))
    df['complete_vehicle_infos'] = df['complete_vehicle_infos'].apply(ast.literal_eval)
    df['fco_vehicle_infos'] = df['fco_vehicle_infos'].apply(ast.literal_eval)
    df['detected_vehicle_infos'] = df['detected_vehicle_infos'].apply(ast.literal_eval)

    # import the config from dataset_path.configs.config.FCO import MULTI_FCO_INTERSECTION
    config_path = os.path.join(dataset_path, 'configs', 'config_FCO.py')
    config = importlib.import_module(config_path.replace('/', '.').replace('.py', ''))
    config = getattr(config, 'MULTI_FCO_INTERSECTION')

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f'Create FCO target for {dataset_path}'):
        current_complete_vehicles = df.iloc[index]['complete_vehicle_infos']
        current_fco_vehicles = df.iloc[index]['fco_vehicle_infos']
        current_detected_vehicles = df.iloc[index]['detected_vehicle_infos']

        # extract the undetected vehicles = complte - detected - fco
        undetected_vehicles = {
            key: value
            for key, value in current_complete_vehicles.items()
            if key not in current_detected_vehicles or key not in current_fco_vehicles
        }
        print(f'undetected_vehicles: {undetected_vehicles.keys()}')
        print(f'current_detected_vehicles: {current_detected_vehicles.keys()}')
        print(f'current_fco_vehicles: {current_fco_vehicles.keys()}')
    
        fig = plot_colored_boxes(current_fco_vehicles, current_detected_vehicles, undetected_vehicles, config['RADIUS'])
        plt.draw()
        fig.savefig(f'{dataset_path}/fco_visualization/{row.iloc[0]}')
        plt.close(fig)


def plot_colored_boxes(fco_vehicles, detected_vehicles, undetected_vehciles, radius):
    # create figure with high resolution
    fig, ax = plt.subplots(figsize=(20, 20), dpi=100)

    intersection_center = [6728.8, 5315.08]

    width = 1.8
    length = 4.5

    ego_info =[0,0,0]

    for box in undetected_vehciles:
        # Create a polygon
        undetected_vehciles[box]['pos_x'] = undetected_vehciles[box]['pos_x'] - intersection_center[0]
        undetected_vehciles[box]['pos_y'] = undetected_vehciles[box]['pos_y'] - intersection_center[1]
        polygon = create_box(width, length,
                             undetected_vehciles[box]['pos_x'], undetected_vehciles[box]['pos_y'],
                             undetected_vehciles[box]['angle'], ego_info)
        polygon.set_facecolor('red')
        polygon.set_edgecolor('red')
        # Add the polygon to the axis
        ax.add_patch(polygon)
    
    for box in detected_vehicles:
        # Create a polygon
        detected_vehicles[box]['pos_x'] = detected_vehicles[box]['pos_x'] - intersection_center[0]
        detected_vehicles[box]['pos_y'] = detected_vehicles[box]['pos_y'] - intersection_center[1]
        polygon = create_box(width, length,
                             detected_vehicles[box]['pos_x'], detected_vehicles[box]['pos_y'],
                             detected_vehicles[box]['angle'], ego_info)
        polygon.set_facecolor('green')
        polygon.set_edgecolor('green')
        # Add the polygon to the axis
        ax.add_patch(polygon)

    for box in fco_vehicles:
        # Create a polygon
        fco_vehicles[box]['pos_x'] = fco_vehicles[box]['pos_x'] - intersection_center[0]
        fco_vehicles[box]['pos_y'] = fco_vehicles[box]['pos_y'] - intersection_center[1]
        polygon = create_box(width, length,
                             fco_vehicles[box]['pos_x'], fco_vehicles[box]['pos_y'],
                             fco_vehicles[box]['angle'], ego_info)
        polygon.set_facecolor('blue')
        polygon.set_edgecolor('blue')
        # Add the polygon to the axis
        ax.add_patch(polygon)

    

    
        # Set the limits of the plot based on the coordinates of the points
    # Adjust the values according to your needs
    plt.xlim(-radius, radius)
    plt.ylim(-radius, radius)
    # Set aspect ratio to be equal
    ax.set_aspect('equal')

    # Hide the axis
    ax.axis('off')

    return fig

def create_vis_video(dataset_path: str, start_time: int, video_length: int):
    """
    This function will generate a video out of the FCO visualization images.
    """
    # check if the vis images got created already
    if not os.path.exists(os.path.join(dataset_path, 'fco_visualization')):
        generate_fco_visualization(dataset_path)
    
    end_time = start_time + video_length
    
    # create the list of images for the video
    image_paths = [f'{dataset_path}/fco_visualization/cluster_25579770_2633530003_2633530004_2633530005_{time}.png' for time in range(start_time, end_time)]

    first_image = cv2.imread(image_paths[0])
    height, width, layers = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID' or 'MJPG'
    video = cv2.VideoWriter(f'{dataset_path}/video.mp4', fourcc, 2, (width, height))

    # Iterate through the image paths and add each image to the video
    for image_path in image_paths:
        img = cv2.imread(image_path)
        video.write(img)

    # Release the video object
    video.release()



if __name__ == '__main__':
    generate_fco_visualization('data/i3040_10pct_6-7h_75m')
    create_vis_video('data/i3040_10pct_6-7h_75m', 23599, 100)