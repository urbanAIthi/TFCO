

from PIL import Image, ImageDraw
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib
import torch


from typing import List

import numpy as np

def get_intersection_image_from_vehicleids(intersection: str, intersections_dict: dict, plot_vehicle_dict: dict, radius: int, target_size: int=400) -> matplotlib.figure.Figure:
    # get center point of intersection
    intersection_pos = [intersections_dict[intersection]['x_pos'], intersections_dict[intersection]['y_pos'], 0]
    img = plot_boxes(plot_vehicle_dict, radius, intersection_pos, vehicle_size=[1.6,3.8], img=True, target_size=target_size)
    return img

def plot_boxes(vehicles: dict, radius: int, ego_info: List[float], vehicle_size: List[float], target_size: int = 400, img: bool = False, figsize: int=100) -> torch.Tensor:
    # Create a figure and axis
    if img:
        fig, ax = plt.subplots(figsize=(target_size/figsize, target_size/figsize))
    else:
        fig, ax = plt.subplots()

    for box in vehicles:
        # Create a polygon
        vehicles[box]['width'] = vehicle_size[0]
        vehicles[box]['length'] = vehicle_size[1]
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



def create_box(width, length, x_pos, y_pos, angle, ego_info: List[float]=[0, 0, 0]):
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


# 'def plot_boxes(vehicles: dict, radius: int, ego_info: List[float] = [0, 0, 0], intersection_center: List[float] = [0, 0], vehicle_size=[1.6, 3.8]):
#     # Create a figure and axis
#     fig, ax = plt.subplots()

#     for box in vehicles:
#         # Create a polygon
#         vehicles[box]['pos_x'] = vehicles[box]['pos_x'] - intersection_center[0]
#         vehicles[box]['pos_y'] = vehicles[box]['pos_y'] - intersection_center[1]
#         polygon = create_box(vehicle_size[0], vehicle_size[1],
#                              vehicles[box]['pos_x'], vehicles[box]['pos_y'],
#                              vehicles[box]['angle'], ego_info)

#         # Add the polygon to the axis
#         ax.add_patch(polygon)

#     # Set the limits of the plot based on the coordinates of the points
#     # Adjust the values according to your needs
#     plt.xlim(-radius, radius)
#     plt.ylim(-radius, radius)
#     # Set aspect ratio to be equal
#     ax.set_aspect('equal')

#     # Hide the axis
#     ax.axis('off')

#     return fig


# def create_box_pillow(width, length, x_pos, y_pos, angle):
#     # Define vertices of the box
#     vertices = np.array([
#         [-width / 2, -length / 2],
#         [width / 2, -length / 2],
#         [width / 2, length / 2],
#         [-width / 2, length / 2]
#     ])

#     # Calculate angle in radians
#     angle_rad = np.radians(-angle)

#     # Create rotation matrix
#     rotation_matrix = np.array([
#         [np.cos(angle_rad), -np.sin(angle_rad)],
#         [np.sin(angle_rad), np.cos(angle_rad)]
#     ])

#     # Rotate vertices using rotation matrix
#     rotated_vertices = np.dot(vertices, rotation_matrix.T)

#     # Translate rotated vertices by position of the box
#     translated_vertices = rotated_vertices + np.array([x_pos, y_pos])

#     # Convert numpy array to list of tuples
#     box_coords = list(map(tuple, translated_vertices))

#     return box_coords'


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