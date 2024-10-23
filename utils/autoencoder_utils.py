import PIL
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import shutil
from typing import List, Tuple, Union

def prepare_datasets(dataset_path: str, dataset_name: str, sequence_len: Union[int, None], train_ids: List[int],
                     val_ids: List[int], test_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
    # check if the dataset was prepared for the given sequence length
    
    if sequence_len is None:
        datasets_in_path = os.listdir(f'data/{dataset_path}')
        train_IDs = [f"data/{dataset_path}/{dataset_path}_{i}.pkl" for i in train_ids]
        # check the train IDs exist
        if not all([os.path.exists(train_ID) for train_ID in train_IDs]):
            raise ValueError(f'The train IDs {train_IDs} do not exist')
        val_IDs = [f"data/{dataset_path}/{dataset_path}_{i}.pkl" for i in val_ids]
        # check the val IDs exist
        if not all([os.path.exists(val_ID) for val_ID in val_IDs]):
            raise ValueError(f'The val IDs {val_IDs} do not exist')
        test_IDs = [f"data/{dataset_path}/{dataset_path}_{i}.pkl" for i in test_ids]
        # check the test IDs exist
        if not all([os.path.exists(test_ID) for test_ID in test_IDs]):
            raise ValueError(f'The test IDs {test_IDs} do not exist')
        return train_IDs, val_IDs, test_IDs
    else:
        datasets_in_path = os.listdir(f'{dataset_path}/{dataset_name}')
        # check if at least one file in the dataset path contains the sequence length
        if not any([f'merged_sequence{sequence_len}' in dataset for dataset in datasets_in_path]):
            raise ValueError(f'The dataset in {dataset_path} was not prepared for the given sequence length {sequence_len}')
        # create the dataset paths
        train_IDs = [f"{dataset_path}/{dataset_name}/merged_sequence{sequence_len}_{dataset_name}_{i}.pkl" for i in train_ids]
        # check the train IDs exist
        if not all([os.path.exists(train_ID) for train_ID in train_IDs]):
            raise ValueError(f'The train IDs {train_IDs} do not exist')
        val_IDs = [f"{dataset_path}/{dataset_name}/merged_sequence{sequence_len}_{dataset_name}_{i}.pkl" for i in val_ids]
        # check the val IDs exist
        if not all([os.path.exists(val_ID) for val_ID in val_IDs]):
            raise ValueError(f'The val IDs {val_IDs} do not exist')
        test_IDs = [f"{dataset_path}/{dataset_name}/merged_sequence{sequence_len}_{dataset_name}_{i}.pkl" for i in test_ids]
        # check the test IDs exist
        if not all([os.path.exists(test_ID) for test_ID in test_IDs]):
            raise ValueError(f'The test IDs {test_IDs} do not exist')
        return train_IDs, val_IDs, test_IDs

def prepare_path_structure(filename: str, base_path: str, pre_train: bool, temporal: str) -> str:
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    if pre_train:
        if not os.path.exists(os.path.join(base_path, 'pre_train')):
            os.mkdir(os.path.join(base_path, 'pre_train'))
        path = os.path.join(base_path, 'pre_train', filename)
    else:
        if not os.path.exists(os.path.join(base_path, temporal)):
            os.mkdir(os.path.join(base_path, temporal))
        path = os.path.join(base_path, temporal, filename)
    os.mkdir(path)
    os.mkdir(os.path.join(path, 'plots'))
    os.mkdir(os.path.join(path, 'plots', 'train'))
    os.mkdir(os.path.join(path, 'plots', 'val'))
    os.mkdir(os.path.join(path, 'models'))
    os.mkdir(os.path.join(path, 'results'))

    # copy the config file to the path
    shutil.copy('configs/config.py', os.path.join(path, 'config.py'))

    return path

def show_image(tensor, show=False, save_path = None):
    # expects a tensor of shape (1,w,h)
    image = tensor.cpu().clone().detach().numpy().squeeze()
    image = image.clip(0, 1)
    if show:
        PIL.Image.fromarray(np.uint8(image * 255)).show()
    if save_path is not None:
        PIL.Image.fromarray(np.uint8(image * 255)).save(save_path)
    return PIL.Image.fromarray(np.uint8(image * 255))
def show_input_output(image_dict: dict, show=False, save_path = None):
    #input and output are tensors of shape (1,w,h)
    image_list = []
    image_names = []
    for image in image_dict:
        image_list.append(show_image(image_dict[image]))
        image_names.append(image)

    # Create a subplot for each image
    fig, axs = plt.subplots(1, len(image_list), figsize=(60, 15))

    for ax, img, name in zip(axs, image_list, image_names):
        ax.imshow(img, cmap='gray')
        ax.set_title(name)
        ax.axis('off')

    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    return fig

def get_current_output(autoencoder, dataset, idx, device, forward_function = None):
    print(dataset[idx][0].to(device).unsqueeze(0).unsqueeze(0).shape)
    autoencoder.eval()
    if forward_function is None:
        pre_out = autoencoder(dataset[idx][0].to(device).unsqueeze(0).unsqueeze(0))[0].squeeze(0).squeeze(0)
    else:
        pre_out = forward_function(dataset[idx][0].to(device).unsqueeze(0).unsqueeze(0))[0].squeeze(0).squeeze(0)
    autoencoder.train()
    return pre_out

def create_train_gif(path):
    path = os.path.join(path, 'plots')
    # Get a list of all files in the directory
    files = [os.path.join(path, file) for file in os.listdir(path) if
             os.path.isfile(os.path.join(path, file))]

    # Sort the files by their creation time
    sorted_files = sorted(files, key=lambda x: os.path.getctime(x))

    # Create a list to store the image frames
    frames = []

    # Iterate through the sorted file paths
    for file in sorted_files:
        # Open each image file
        image = Image.open(file)

        # Append the image to the frames list
        frames.append(image)

    # Specify the output file path for the GIF
    output_file = os.path.join(path, 'animation.gif')

    # Save the frames as a GIF animation
    frames[0].save(output_file, format='GIF', append_images=frames[1:], save_all=True, duration=50, loop=0)

class MyDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        image = self.data_dict[idx][0]
        return image

if __name__ == "__main__":
    create_train_gif('../trained/autoencoder_2023-06-07_10-45-28')
