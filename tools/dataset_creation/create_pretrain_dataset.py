import os
import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.bev_utils import plot_boxes, plot_boxes_pillow


"""
This file is used to generate arbitrary amount of pre-training data for the Autoencoder-style pre-training of the final spacial-temporal networks.
"""

def does_overlap(pos1: Tuple[float, float], pos2: Tuple[float, float], width: float, length: float) -> bool:
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    return dx < width and dy < length


def generate_non_overlapping_vehicles(num_vehicles: int, radius: int, width: float, length: float) -> dict:
    vehicles = {}
    attempts = 0
    max_attempts = 1000  # Maximum number of attempts to try before giving up

    for i in range(num_vehicles):
        while attempts < max_attempts:
            attempts += 1
            # Randomly generate a position and angle
            pos_x = np.random.uniform(-radius, radius)
            pos_y = np.random.uniform(-radius, radius)
            angle = np.random.uniform(0, 360)

            # Check for overlaps
            overlap = False
            for other_vehicle in vehicles.values():
                if does_overlap((pos_x, pos_y), (other_vehicle['pos_x'], other_vehicle['pos_y']), width, length):
                    overlap = True
                    break
            
            # If no overlap, add the vehicle to the dictionary
            if not overlap:
                vehicles[f'vehicle_{i}'] = {
                    'pos_x': pos_x,
                    'pos_y': pos_y,
                    'angle': angle
                }
                break
        
        if attempts >= max_attempts:
            print("Could not place all vehicles without overlap after maximum attempts.")
            break

    return vehicles

if __name__ == "__main__":
    num_images = 100000
    min_vehicles = 0
    max_vehicles = 30
    radius = 75
    width = 1.8
    length = 4.5
    generator = 'matplotlib' # pillow or matplotlib
    name = f'pre_train_data_{num_images}_{min_vehicles}_{max_vehicles}_{radius}'
    if not os.path.exists('pre_train_data'):
        os.makedirs('pre_train_data')
    if not os.path.exists(name):
        os.makedirs(os.path.join('pre_train_data', name), exist_ok=True)
    for i in tqdm(range(num_images)):
        num_vehicles = random.randint(min_vehicles, max_vehicles)
        vehicles = generate_non_overlapping_vehicles(num_vehicles, radius, width, length)
        if generator == 'pillow':
            fig = plot_boxes_pillow(vehicles, radius)
            fig.save(f'pre_train_data/{name}/{i}.png', dpi = ((100,100)))
        elif generator == 'matplotlib':
            fig = plot_boxes(vehicles, radius)
            plt.draw()
            fig.savefig(f'pre_train_data/{name}/{i}.png')
            plt.close(fig)
        else:
            raise ValueError(f'Unknown generator: {generator}')
    
    