import os
import ast
import importlib
from typing import List
import matplotlib.patches as patches
from PIL import Image, ImageDraw

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import torch
import matplotlib

def plot_boxes(vehicles: dict, radius: int, ego_info: List[float], target_size: int = 400, img: bool = False, figsize: int=100) -> torch.Tensor:
    # Create a figure and axis
    if img:
        fig, ax = plt.subplots(figsize=(2.56, 2.56))
    else:
        fig, ax = plt.subplots()

    for box in vehicles:
        # Create a polygon
        vehicles[box]['width'] = 1.7
        vehicles[box]['length'] = 4
        polygon = create_box(vehicles[box]['width'], vehicles[box]['length'],
                             vehicles[box]['pos_x'], vehicles[box]['pos_y'],
                             vehicles[box]['angle'], ego_info)
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

    if img:
        return fig

    # Show the plot
    # plt.show()
    plt.savefig('tmp/tmp.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Load the saved image
    image = Image.open('tmp/tmp.png')

    # Convert the image to grayscale (single channel)
    gray_image = image.convert('L')

    # Resize the image to 400x400
    resized_image = gray_image.resize((target_size, target_size))

    # Convert the image to a numpy array
    image_array = np.array(resized_image)

    # Convert the numpy array to a Torch tensor
    tensor_image = torch.from_numpy(image_array)

    # Add an additional dimension to represent the single channel
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image

def get_intersection_image_from_vehicleids(intersection: str, intersections_dict: dict, plot_vehicle_dict: dict, radius: int) -> matplotlib.figure.Figure:
    # get center point of intersection
    intersection_pos = [intersections_dict[intersection]['x_pos'], intersections_dict[intersection]['y_pos'], 0]
    img = plot_boxes(plot_vehicle_dict, radius, intersection_pos, img=True)
    return img

def get_empty_plot():
    fig, ax = plt.subplots()
    ax.set_axis_off()  # Hide the axes
    fig.set_facecolor('white')  # Set background color to white
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding and margins
    return fig

def create_box(width, length, x_pos, y_pos, angle, ego_info=None):
    ego_x, ego_y, ego_angle = ego_info

    # Calculate center_point
    center_point = np.array([0, length / 2])

    # Define vertices of the box
    vertices = np.array([
        [-width / 2, 0],
        [width / 2, 0],
        [width / 2, -length],
        [-width / 2, -length]
    ])

    # Calculate angle in radians and adjust with ego's angle
    angle_rad = np.radians(-angle)

    # Create rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Rotate vertices using rotation matrix
    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    rotated_center_point = np.dot(center_point, rotation_matrix.T)

    # Translate rotated vertices by relative position of the box to the ego vehicle
    global_vertices = rotated_vertices + np.array([x_pos, y_pos])
    #print(f'global_vertices: {global_vertices}')

    # Convert global vertices to ego coordinate system
    theta = np.radians(ego_angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])  # rotation matrix

    t = np.array([ego_x, ego_y])  # translation vector

    # Apply transformation
    global_vertices[:, :2] = np.dot(global_vertices[:, :2] - t, R.T)

    # Log the final position of the vertices
    #print(f'global_vertices: {global_vertices}')

    # Create a polygon using the final vertices
    polygon = patches.Polygon(global_vertices, closed=True, fill=True, edgecolor='black', facecolor='black')

    return polygon

def plot_boxes_pillow(vehicles: dict, radius: int, intersection_center: List[float] = [0, 0], image_size=(800,800)):
    # Create a blank image with white background
    img = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(img)

    width = 1.8
    length = 4.5

    # Calculate scale to fit the vehicles within the image dimensions
    scale = min(image_size) / (2 * radius)

    for box in vehicles:
        # Adjust positions based on the intersection center and scale
        pos_x = (vehicles[box]['pos_x'] - intersection_center[0]) * scale + image_size[0] / 2
        pos_y = (vehicles[box]['pos_y'] - intersection_center[1]) * scale + image_size[1] / 2

        # Get the rotated box coordinates
        box_coords = create_box_pillow(width * scale, length * scale, pos_x, pos_y, vehicles[box]['angle'])

        # Draw the polygon on the image
        draw.polygon(box_coords, fill='black')

    return img

def create_box_pillow(width, length, x_pos, y_pos, angle):
    # Define vertices of the box
    vertices = np.array([
        [-width / 2, -length / 2],
        [width / 2, -length / 2],
        [width / 2, length / 2],
        [-width / 2, length / 2]
    ])

    # Calculate angle in radians
    angle_rad = np.radians(-angle)

    # Create rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # Rotate vertices using rotation matrix
    rotated_vertices = np.dot(vertices, rotation_matrix.T)

    # Translate rotated vertices by position of the box
    translated_vertices = rotated_vertices + np.array([x_pos, y_pos])

    # Convert numpy array to list of tuples
    box_coords = list(map(tuple, translated_vertices))

    return box_coords



def create_fco_target(path: str, sequence_len: int):
    # load the csv dataset file into a pandas dataframe
    dataset = pd.read_csv(os.path.join(path, 'dataset.csv')).reset_index(drop=True)
    dataset['intersection'] = dataset['intersection'].apply(ast.literal_eval)
    dataset['complete_vehicle_infos'] = dataset['complete_vehicle_infos'].apply(ast.literal_eval)
    dataset['fco_vehicle_infos'] = dataset['fco_vehicle_infos'].apply(ast.literal_eval)
    dataset['detected_vehicle_infos'] = dataset['detected_vehicle_infos'].apply(ast.literal_eval)
    dataset['intersection_id'] = dataset['intersection'].apply(lambda x: list(x.keys())[0])
    fco_mintime_dict = dict.fromkeys(dataset['intersection_id'].unique().tolist())

    # import the config from dataset_path.configs.config.FCO import MULTI_FCO_INTERSECTION
    config_path = os.path.join(path, 'configs', 'config_FCO.py')
    config = importlib.import_module(config_path.replace('/', '.').replace('.py', ''))
    config = getattr(config, 'MULTI_FCO_INTERSECTION')

    # make dir for the sequence targets
    os.makedirs(os.path.join(path, f'detected_vehicles_time_{sequence_len}'), exist_ok=True)

    # create emtpy series for the detected_vehicles_time_{sequence_len}
    detected_vehicles_time = pd.Series(dtype=object)

    intersection_mintime_dict = dict.fromkeys(dataset['intersection_id'].unique().tolist())
    for key in intersection_mintime_dict.keys():
        intersection_mintime_dict[key] = dataset[dataset['intersection_id'] == key]['time'].min()
    
    for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f'Create detected_vehicles_time_{sequence_len}'):
        current_time = row['time']
        intersection = row['intersection_id']
        sequence_times = torch.arange(current_time, current_time - (sequence_len +1), -1)
        # remove the times that are smaller than the intersection mintime
        sequence_times = sequence_times[sequence_times >= intersection_mintime_dict[intersection]]

        # get the dicts of the vehicles for the sequence times
        time_vehicle_list = list()
        for time in sequence_times:
            idx = dataset[(dataset['time'] == int(time)) & (dataset['intersection_id'] == intersection)].index[0]
            time_vehicle_list.extend(list(dataset['detected_vehicle_infos'][idx].keys()))
        # remove duplicates
        time_vehicle_list = list(set(time_vehicle_list))
        # remove the vehicles that are not in the current frame
        complete_ids = list(row['complete_vehicle_infos'].keys())
        time_vehicle_list = [veh for veh in time_vehicle_list if veh in complete_ids]
        # create the dict for the current time
        detected_time_vehicle_dict = {vehicle: row['complete_vehicle_infos'][vehicle] for vehicle in time_vehicle_list}

        # save the dict in the series
        detected_vehicles_time.at[index] = detected_time_vehicle_dict

        # create plot according to the fco_time_vehicle_dict
        if len(detected_time_vehicle_dict) != 0:
            fig = get_intersection_image_from_vehicleids(intersection, row['intersection'], detected_time_vehicle_dict, config['RADIUS'])
            plt.draw()
            fig.savefig(os.path.join(path, f'detected_vehicles_time_{sequence_len}', f'{row["id"]}.png'), bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            img = get_empty_plot()
            plt.draw()
            img.savefig(os.path.join(path, f'detected_vehicles_time_{sequence_len}', f'{row["id"]}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()
    
    #add the detected_vehicles_time_{sequence_len} to the dataset
    dataset[f'detected_time_{sequence_len}_vehicle_infos'] = detected_vehicles_time

    # save the dataframe with the new column containing the fco_time_dict information for the current sequence_length without the first row
    dataset.to_csv(os.path.join(path, 'dataset.csv'), index=False)


if __name__ == "__main__":
    from configs.config import special_temporal_decoder_new as cfg
            # create the dataset target according to the current dataset if it does not exist
    if not os.path.exists(f'data/{cfg["dataset_name"]}/fco_time_{cfg["sequence_len"]}'):
        create_fco_target(f'data/{cfg["dataset_name"]}', cfg['sequence_len'])
    else:
        print(f'fco target already exists for {cfg["dataset_name"]} with sequence length {cfg["sequence_len"]}')
    print('done')