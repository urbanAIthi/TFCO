import os
import pickle
import tempfile
import numpy as np
import torch
from typing import List, Union
from einops import rearrange
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from utils.bev_utils import plot_boxes
from PIL import Image
from torchvision import transforms
import tqdm as tqdm
from scipy.ndimage import label, find_objects

from utils.train_utils import unmerge_sequence


def compute_iou(eval_prob: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5):
    # Convert sigmoid output to binary mask using threshold
    eval_mask = (eval_prob >= threshold).int()
    gt_mask = (gt_mask >= threshold).int()

    # Invert the masks
    eval_mask = 1 - eval_mask
    gt_mask = 1 - gt_mask

    # Compute the intersection
    intersection = torch.sum(eval_mask & gt_mask)

    # Compute the union
    union = torch.sum(eval_mask) + torch.sum(gt_mask) - intersection

    if union == 0:
        return None
    else:
        iou = intersection.float() / union.float()

    return iou


def get_mean_iou(output_batch: torch.tensor, target_batch: torch.tensor, threshold: float = 0.5):
    iou = 0
    counter = 0
    for output, target in zip(output_batch, target_batch):
        current_iou = compute_iou(output, target)
        if current_iou is not None:
            counter += 1
            iou += current_iou
    if counter == 0:
        return 0
    else:
        return iou / counter

def show_image(image: torch.tensor, save_path: str = 'test.png'):
    """
    This function shows a singe image that currently is a tensor as a matplotlib plot
    :param image: the tensor that should be shown
    :param save_path: the path where the image should be saved
    """

    # convert the image to a numpy array
    image = image.detach().cpu().numpy()
    # change the shape of the image
    image = np.transpose(image, (1, 2, 0))
    # show the image
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.savefig(save_path)
    plt.show()

def get_single_vehicle_fombatchdict(vehicle_dict: dict, index: int):
    single = list(vehicle_dict[index].items())

    key = single[0][0]
    value = single[0][1]

    # Create a new dictionary with this single key-value pair
    new_dict = {key: value}

    return new_dict

def get_unhidden_vehicles(output_batch: torch.tensor, gt_positions_batch: List[dict], current_positions_batch: List[dict], input_batch: torch.tensor, pixel_threshold: float = 0.5, vehicle_threshold: float = 0.5) -> dict:
    """
    The goal of this function is to analyze the number of vehicles that got correctly unhidden by the tempporal part of the model. 
    :param output_batch: the output of the model
    :param gt_positions: the ground truth positions of the vehicles from the sumo simulation
    :param input_batch: the input sequence of the model
    :param pixel_threshold: the threshold for the pixel values are counted as part of a vehicle
    :param vehicle_threshold: the threshold for the percentage of pixels that need to be active for the vehicle to be counted as unhidden
    """
    # change the shape of the input batch such that batch is fist
    input_batch = rearrange(input_batch, 's b c h w -> b s c h w')
    # iterate over the batch
    for output, input, gt_positions, current_positions in zip(output_batch, input_batch, gt_positions_batch, current_positions_batch):

        # get the last image of the input sequence
        input = input[-1, ...]

        # get vehicles that are seen through the temporal model
        ## get the center points of the gt positions within the image
        gt_vehicle_pixels_list = get_vehicle_pixels_new(gt_positions)
        ## check which of the gt positions are within the image
        gt_overlap_count, gt_nonoverlap_count = get_overlaps_frompixels(output, gt_vehicle_pixels_list)
        ## get the center points of the input positions within the image
        current_vehicle_pixels_list = get_vehicle_pixels_new(current_positions)
        input_overlap_count, input_nonoverlap_count = get_overlaps_frompixels(output, current_vehicle_pixels_list)
        
        ## check which of the input positions are within the image
        ## --> get lost vehicles from input
        lost_vehicles = input_nonoverlap_count
        ## --> get newly seen vehicles through gt
        new_vehicles = gt_overlap_count - input_overlap_count
        ## --> get the max new vehicles
        max_new_vehicles = len(gt_positions) - len(current_positions)

        print(f'lost vehicles: {lost_vehicles}, new vehicles: {new_vehicles}, max new vehicles: {max_new_vehicles}')

        # get vehicles that are falsely unhidden
        ## create ((output -  GT) + 1) and map back to [0,1]
        ## --> check if any vehicles are falsely unhidden - will be black in the created image
    
    return lost_vehicles, new_vehicles

def get_centerpoints_fromdict(vehicles: dict):
    center_points = []
    
    intersection_center = [6728.8, 5315.08]

    width = 1.8
    length = 4.5

    image_width = 400
    image_height = 400

    radius = 75

    original_width, original_height = 2*radius, 2*radius

    for box in vehicles:
        center_x = vehicles[box]['pos_x']
        center_y = vehicles[box]['pos_y'] - length / 2
        center_points.append((center_x, center_y))

        # correct the center point according to the intersection center
        center_x -= intersection_center[0]
        center_y -= intersection_center[1]

        # rotate the center point
        angle_rad = np.radians(-vehicles[box]['angle'])
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        # Adjust the center point for rotation
        # Note: This assumes the rotation is around the center-top point
        rotated_center = np.dot(rotation_matrix, np.array([center_x - vehicles[box]['pos_x'], center_y - vehicles[box]['pos_y']]))
        rotated_center += np.array([vehicles[box]['pos_x'], vehicles[box]['pos_y']])

        center_points.append((rotated_center[0], rotated_center[1]))

    pixel_centers = []
    for cx, cy in center_points:
        # Transform to pixel coordinates
        pixel_x = (cx + radius) / original_width * image_width
        pixel_y = image_height - (cy + radius) / original_height * image_height
        pixel_centers.append((pixel_x, pixel_y))
    
    return pixel_centers

def get_vehicle_pixels_new(vehicles:dict, radius: int=75, ego_info: List[float] = [0, 0, 0], intersection_center: List[float] = [6728.8, 5315.08], transform=transforms.Compose([
                                            transforms.Resize((400,400)),
                                            transforms.ToTensor(),  # Convert the PIL Image to a PyTorch tensor
                                        ])):
    pixels_list  = []
    for id, vehicle in enumerate(vehicles):
        vehicle_dict = vehicles[vehicle]
        # check if the value sin vehciel dict are negative --> if so, make them positive
        for key in vehicle_dict:
            if vehicle_dict[key] < 0:
                vehicle_dict[key] = abs(vehicle_dict[key])
        fig = plot_boxes({vehicle: vehicle_dict}, 75)
        # temporally save the figure
        plt.draw()
        fig.savefig('eval.png')
        plt.close(fig)
        # read the image using PIL (as in the dataloder)
        image = Image.open('eval.png').convert('L')
        # transform the image
        image = transform(image)
        # create binary map 
        image = (image > 0.5).int()
        # invert the image --> vehicles get 1, background gets 0
        image = 1 - image
        # get a list of pixel coordinates that are part of the vehicle
        vehicle_pixels = torch.nonzero(image)
        # check if the vehicle pixels is empty
        if len(vehicle_pixels) == 0:
            continue
        else:
            pixels_list.append(vehicle_pixels)
    return pixels_list







def get_vehicle_pixels(vehicles: dict, radius: int=75, ego_info: List[float] = [0, 0, 0], intersection_center: List[float] = [6728.8, 5315.08]):
    vehicle_pixels = []

    width = 1.8
    length = 4.5
    image_width, image_height = 400, 400
    original_width, original_height = 2 * radius, 2 * radius

    for box in vehicles:
        # Adjust position relative to the intersection center
        x_pos = vehicles[box]['pos_x'] - intersection_center[0]
        y_pos = vehicles[box]['pos_y'] - intersection_center[1]
        angle = vehicles[box]['angle']

        # Calculate vertices of the box
        vertices = np.array([
            [-width / 2, 0],
            [width / 2, 0],
            [width / 2, -length],
            [-width / 2, -length]
        ])

        # Apply rotation
        angle_rad = np.radians(-angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        rotated_vertices = np.dot(vertices, rotation_matrix.T)

        # Translate rotated vertices by relative position
        global_vertices = rotated_vertices + np.array([x_pos, y_pos])

        # Apply ego vehicle transformation
        ego_x, ego_y, ego_angle = ego_info
        theta = np.radians(ego_angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        t = np.array([ego_x, ego_y])
        global_vertices[:, :2] = np.dot(global_vertices[:, :2] - t, R.T)

        # Transform vertices to pixel coordinates
        pixel_vertices = [(int((x + radius) / original_width * image_width), 
                        int((y + radius) / original_height * image_height)) for x, y in global_vertices]

        # Create an empty mask
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # Fill the polygon in the mask
        cv2.fillPoly(mask, [np.array(pixel_vertices, dtype=np.int32)], 1)

        # Extract the pixel coordinates from the mask
        y_coords, x_coords = np.where(mask == 1)
        vehicle_pixels.append(list(zip(x_coords, y_coords)))

    return vehicle_pixels

def get_vehicle_pixels_matplotlib(vehicles, radius=75, ego_info=[0, 0, 0], intersection_center=[6728.8, 5315.08]):
    vehicle_pixels = []
    image_width, image_height = 400, 400
    original_width, original_height = 2 * radius, 2 * radius
    width = 1.8
    length = 4.5

    for box in vehicles:
                # Adjust position relative to the intersection center
        x_pos = vehicles[box]['pos_x'] - intersection_center[0]
        y_pos = vehicles[box]['pos_y'] - intersection_center[1]
        angle = vehicles[box]['angle']

        # Calculate vertices of the box
        vertices = np.array([
            [-width / 2, 0],
            [width / 2, 0],
            [width / 2, -length],
            [-width / 2, -length]
        ])

        # Apply rotation
        angle_rad = np.radians(-angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        rotated_vertices = np.dot(vertices, rotation_matrix.T)

        # Translate rotated vertices by relative position
        global_vertices = rotated_vertices + np.array([x_pos, y_pos])

        # Apply ego vehicle transformation
        ego_x, ego_y, ego_angle = ego_info
        theta = np.radians(ego_angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        t = np.array([ego_x, ego_y])
        global_vertices[:, :2] = np.dot(global_vertices[:, :2] - t, R.T)

        # Transform vertices to pixel coordinates
        pixel_vertices = [(int((x + radius) / original_width * image_width), 
                        int((y + radius) / original_height * image_height)) for x, y in global_vertices]

        # Transform vertices to pixel coordinates
        pixel_vertices = [(int((x + radius) / original_width * image_width), 
                           int((y + radius) / original_height * image_height)) for x, y in global_vertices]

        # Create a Path object from the vertices
        path = Path(pixel_vertices)
        patch = patches.PathPatch(path, facecolor='none', lw=0)

        # Create an empty mask
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # Create a figure and add the patch to it
        fig, ax = plt.subplots()
        ax.add_patch(patch)
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        ax.axis('off')

        # Draw the figure to a buffer
        fig.canvas.draw()

        # Convert the buffer to an array and extract the mask
        mask = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        # Extract the pixel coordinates from the mask
        y_coords, x_coords = np.where(mask[:, :, 0] != 0)  # Assuming non-zero values mark the vehicle
        vehicle_pixels.append(list(zip(x_coords, y_coords)))

    return vehicle_pixels


def get_overlaps_frompixels(output: torch.tensor, all_vehicle_pixels: List, pixel_threshold: float=0.5, vehicle_threshold: float=0.75):
    overlap_count = 0
    nonoverlap_count = 0

    # map the pixels of the output image to the threshold and invert the image --> vehicles get 1, background gets 0
    output_mask = (output >= pixel_threshold).int().detach().cpu()

    # invert the mask
    output_mask = 1 - output_mask

    # get the non zero pixels
    output_mask_nonzero = torch.nonzero(output_mask)

    for vehicle_pixels in all_vehicle_pixels:
        counter = 0
        for vehicle_pixel in vehicle_pixels:
            if vehicle_pixel in output_mask_nonzero:
                counter += 1
        if counter / len(vehicle_pixels) >= vehicle_threshold:
            overlap_count += 1
        else:
            nonoverlap_count += 1
    
    return overlap_count, nonoverlap_count

def get_overlaps_fromcenterpoints(output: torch.tensor, image_center_points: List, pixel_threshold: float, n_pixels: int):    
    overlap_count = 0
    nonoverlap_count = 0

    channel, height, width = output.shape

    for center in image_center_points:
        x,y = center
        lower_x, upper_x = max(0, x - n_pixels), min(width, x + n_pixels + 1)
        lower_y, upper_y = max(0, y - n_pixels), min(height, y + n_pixels + 1)

        surrounding_pixels = output[lower_y:upper_y, lower_x:upper_x]

        # Check if all surrounding pixels are below the threshold
        if torch.all(surrounding_pixels < pixel_threshold):
            overlap_count += 1
        
        else:
            nonoverlap_count += 1
    
    return overlap_count, nonoverlap_count

def get_num_halocinate(output_mask: torch.tensor, gt_mask: torch.tensor, min_object_pixels: int = 35):
    # first get the diff mask in order to delete the correctly detected vehicles and bring back to [0,1]
    diff_mask = output_mask - gt_mask
    # Convert the tensor to a NumPy array and squeeze it to 2D
    mask_np = diff_mask.squeeze().numpy()

    # Label connected components
    labeled_array, num_features = label(mask_np)

    # Filter out small objects
    object_slices = find_objects(labeled_array)
    for object_slice in object_slices:
        if np.sum(labeled_array[object_slice]) < min_object_pixels:
            labeled_array[object_slice] = 0

    # Recount the number of objects after filtering
    unique_objects = np.unique(labeled_array)
    unique_objects = unique_objects[unique_objects != 0]  # Exclude background
    object_count = len(unique_objects)

    return object_count


def analyze_results(outputs: torch.Tensor, target_sequence: torch.Tensor, input_sequence: torch.Tensor,
                     time: int, epoch: int, path: str, pre_train: bool = False, show: bool =False, save_plot = False, save_data = False,
                     overlay_image_path: Union[str, None] = None, overlay_alpha: float = 0.4):
    
    if pre_train:
        input_sequence = input_sequence[-1:, ...]
        target_sequence = target_sequence[-1:, ...]
    input_sequence = unmerge_sequence(input_sequence, 'biv_t')
    target_sequence = unmerge_sequence(target_sequence, 'biv_target_t')

    if overlay_image_path is not None:
        overlay_image = Image.open(overlay_image_path)

    # check if the outputs already run through a sigmoid function
    if torch.max(outputs) > 1:
        outputs = torch.sigmoid(outputs)

    # create matplotlib plot of the input sequence in first row and the target sequence in the second row
    fig, axs = plt.subplots(3, len(input_sequence) +1, figsize=(40, 20))
    for i, ((key, value), (target_key, target_value)) in enumerate(zip(input_sequence.items(), target_sequence.items())):
        axs[0, i].imshow(value[0], cmap='gray')
        axs[0, i].set_title(key)
        axs[0, i].axis('off')
        axs[2, i].imshow(target_value[0], cmap='gray')
        axs[2, i].set_title(target_key)
        axs[2, i].axis('off')

        if overlay_image_path is not None:
            # Resize the overlay image to match the size of the image_array and then overlay it
            resized_overlay = overlay_image.resize((value[0].shape[1], value[0].shape[0]))
            axs[0, i].imshow(np.array(resized_overlay), alpha=overlay_alpha)
            axs[1, i].imshow(np.array(resized_overlay), alpha=overlay_alpha)

    # add the output of the model to the plot
    axs[1, len(input_sequence)-1].imshow(outputs[0].detach().cpu().numpy(), cmap='gray')
    axs[1, len(input_sequence)-1].set_title('output')
    axs[1, len(input_sequence)-1].axis('off')

    # Fill unused axes with white and turn off their axes
    for ax in axs.flatten():
        if not ax.has_data():
            ax.set_facecolor('white')
            ax.set_axis_off()

    if save_plot:
        plt.savefig(os.path.join(path, f'epoch_{epoch}_time_{time}.png'))

    if show:
        plt.show()

    if save_data:
        # save the input_sequence, target_sequence and output_sequence as pickle file
        with open(os.path.join(path, f'epoch_{epoch}_time_{time}.pkl'), 'wb') as file:
            pickle.dump({'input_sequence': input_sequence, 'target_sequence': target_sequence, 'output_sequence': outputs}, file)

        # Log plot to wandb
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file.name, format='png')
        #wandb.log({"plot": wandb.Image(temp_file.name, caption=f"Plot at epoch {epoch}, time {time[index]}")}, step=epoch)
        os.unlink(temp_file.name)  # Remove the temporary file

    plt.close(fig)


        

