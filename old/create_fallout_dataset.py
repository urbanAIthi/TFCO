import pandas as pd
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import PIL
from utils.autoencoder_utils import show_image
import torch
import os
from typing import Dict, Union
from configs.configs_seqdataset import SEQ_DATASET

'''
This file defines the functions to create a dataset for the spacial temporal decoder.
'''

def create_fallout_dataset(data_fallout: Dict[str, Dict[str, Dict]], data_nofallout: Dict[str, Dict[str, torch.Tensor]], filename: str,dataset_name: str, sequ_len: int):
    '''
    '''
    # get rid of the floating point keys
    data_fallout = {int(k): data_fallout[k] for k in list(data_fallout.keys())}
    data_nofallout = {int(k): data_nofallout[k] for k in list(data_nofallout.keys())}
    # set all image tesnors to 0 to test functionality
    for key in data_fallout.keys():
    #    data_fallout[key]['image_tensor'] = 0
        data_fallout[key]['time'] = key
    #for key in data_nofallout.keys():
    #    data_nofallout[key]['image_tensor'] = 0
    # create a dataframe from the dictionary
    df_fallout = pd.DataFrame.from_dict(data_fallout).T
    df_nofallout = pd.DataFrame.from_dict(data_nofallout).T
    # rename the columns
    df_fallout.rename(columns={'image_tensor': 'biv_t'}, inplace=True)
    df_fallout.rename(columns={'vehicle_ids': 'detected_vehicles'}, inplace=True)
    df_nofallout.rename(columns={'image_tensor': 'biv_target_t'}, inplace=True)
    df_nofallout.rename(columns={'vehicle_ids': 'all_vehicles'}, inplace=True)
    # Add the target column
    df_fallout['biv_target_t'] = df_nofallout['biv_target_t']
    df_fallout['all_vehicles'] = df_nofallout['all_vehicles']
    # Create additional columns with shifted values
    for i in range(1, sequ_len+1):
        df_fallout[f'biv_t-{i}'] = df_fallout['biv_t'].shift(i)
        df_fallout[f'detected_vehicles-{i}'] = df_fallout['detected_vehicles'].shift(i)
        df_fallout[f'time-{i}'] = df_fallout['time'].shift(i)
        df_fallout[f'all_vehicles-{i}'] = df_fallout['all_vehicles'].shift(i)
        df_fallout[f'biv_target_t-{i}'] = df_fallout['biv_target_t'].shift(i)
    # Delete the first -sequence_len- rows since they contain NaN values
    df_nofallout = df_nofallout.drop(df_nofallout.index[0:sequ_len])
    df_fallout = df_fallout.drop(df_fallout.index[0:sequ_len])
    df_dataset = df_fallout



    dataset_dict = df_dataset.to_dict(orient='index')

    # clean the dataset such that no black images are in the dataset
    for key in list(dataset_dict.keys()):
        # iterate over the dict of the current key that have "biv" at the beginning
        for key2 in list(dataset_dict[key].keys()):
            if key2.startswith('biv'):
                # check if the image is black
                if dataset_dict[key][key2].max() == 0:
                    # remove the key from the dict
                    del dataset_dict[key]
                # print the mean of the image
                #print(dataset_dict[key][key2].float().mean())


    with open(f'data/{dataset}/sequence{sequ_len}_{filename}.pkl', 'wb') as f:
        pickle.dump(dataset_dict, f)

def show_dataset_sample(dataset_path: str, target_image_path: str = 'example_sequence.png', overlay_image_path: Union[str, None] = None, overlay_alpha: float = 0.4):
    #reduce dataset_name to the pikle file name only
    dataset_name = dataset_path.split('/')[-1]
    # get the sequence length from the dataset name (int value between 'sequence' and '_')
    sequence_len = int(dataset_name.split('_')[0].split('sequence')[1])
    with open(dataset_path, 'rb') as f:
        dataset= pickle.load(f)
    # get random key from the dataset
    index = np.random.choice(list(dataset.keys()))
    try:
        image_dict = dataset[index]
    except:
         raise ValueError (f'Index out of range index needs to be in range {list(dataset.keys())[0]} - {list(dataset.keys())[-1]}')
    # Create a figure to display the images
    fig, axs = plt.subplots(2, sequence_len+1, figsize=(25, 13))
    # remove the 'vehicle_ids' key from the dictionary if it exists
    if 'vehicle_ids' in image_dict:
        image_dict.pop('vehicle_ids')
    # get the imag_dict keys that start with biv_t and do not contain target in the key string
    img_dict = {key: image_dict[key] for key in image_dict.keys() if key.startswith('biv_t') and 'target' not in key}
    img_dict = dict(reversed(img_dict.items()))
    target_dict = {key: image_dict[key] for key in image_dict.keys() if key.startswith('biv_target_t')}
    target_dict = dict(reversed(target_dict.items()))

    # load the overlay image if path is not None
    if overlay_image_path is not None:
        overlay_image = PIL.Image.open(overlay_image_path)

    # Iterate over the img_dict and target_dict and add them to the plot
    for i, ((image_key, image_tensor), (target_key, target_tensor))  in enumerate(zip(img_dict.items(), target_dict.items())):
        # Convert the torch tensor to a NumPy array
        image_array = image_tensor.squeeze().numpy()
        image_array = image_array.astype(np.uint8)
        # Show the image using Matplotlib
        axs[0,i].imshow(image_array, cmap='gray')
        axs[0,i].set_title(image_key)
        axs[0,i].axis('off')

        target_array = target_tensor.squeeze().numpy()
        target_array = target_array.astype(np.uint8)
        # Show the image using Matplotlib
        axs[1,i].imshow(target_array, cmap='gray')
        axs[1,i].set_title(target_key)
        axs[1,i].axis('off')

        if overlay_image_path is not None:
            # Resize the overlay image to match the size of the image_array and then overlay it
            resized_overlay = overlay_image.resize((image_array.shape[1], image_array.shape[0]))
            axs[0, i].imshow(np.array(resized_overlay), alpha=overlay_alpha) 
            axs[1, i].imshow(np.array(resized_overlay), alpha=overlay_alpha)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the figure
    plt.savefig(target_image_path)


if __name__ == '__main__':
    dataset = SEQ_DATASET['DATASET_NAME']
    # iterate over the .pkl files in the dataset
    for file in os.listdir(os.path.join('data',dataset)):
        # check if the file is a .pkl file and does not start with 'sequence'
        if file.endswith('.pkl') and not file.startswith('sequence'):
            print(f'creating dataset for {file}')
            # load the file
            with open(os.path.join('data',dataset, file), 'rb') as f:
                data = pickle.load(f)
            # create the dataset
            filename = file.split('.')[0]
            create_fallout_dataset(data[SEQ_DATASET['INTERSECTION']]['Fallout'], data[SEQ_DATASET['INTERSECTION']]['NoFallout'], filename, dataset, SEQ_DATASET['SEQU_LEN'])

