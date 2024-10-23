import pandas as pd
import numpy as np
import os
import torch
import random
from PIL import Image
import ast
import tqdm






def create_nonseen_dataset(dataset_path, sequence_len):
    # open the dataset.csv 
    print('Opening dataset.csv')
    dataset = pd.read_csv(os.path.join(dataset_path, 'dataset.csv'))
    nonseen_dataset = dataset.copy()
    print('Dataset opened')
    # add another column to the dataset


    # iterate throught the dataset
    for index, row in tqdm.tqdm(dataset.iterrows(), total=len(dataset)):
        fco_vehicle_infos = ast.literal_eval(row['fco_vehicle_infos'])
        try: 
            time_vehicle_infos = ast.literal_eval(row[f'fco_time_dict_{sequence_len}'])
        except: 
            time_vehicle_infos = {}
        detected_vehicles = ast.literal_eval(row['detected_vehicle_infos'])
        if len(fco_vehicle_infos) != 0:
            current_seen_vehicles = list(set(list(fco_vehicle_infos.keys()) + list(detected_vehicles.keys())))
            time_seen_vehicles = list(time_vehicle_infos.keys())
            diff = len(time_seen_vehicles) - len(current_seen_vehicles)
            # add diff value to the dataset
            nonseen_dataset.at[index, f'diff_{sequence_len}'] = diff
        #current_image_name = f'{row["Unnamed: 0"]}.png'
        # open the bev_image
        #bev_image = Image.open(os.path.join(dataset_path, 'bev_images', current_image_name))

        # open the fco_time_{sequence_len}
        #fco_time = Image.open(os.path.join(dataset_path, 'fco_time', f'fco_time_{sequence_len}', current_image_name))

        # compare if fco_time and bev_image are identical
        #if torch.all(torch.eq(torch.tensor(bev_image), torch.tensor(fco_time))):
        #    # if they are identical, remove the row from the dataset
        #    nonseen_dataset.drop(index=index, inplace=True)
        

    # save the new dataset in the same folder
    new_dataset_path = os.path.join(dataset_path, f'dataset.csv')
    dataset.to_csv(new_dataset_path, index=False)


if __name__ == '__main__':
    create_nonseen_dataset('data/i3040_10pct_6-9h_75m_large', 5)