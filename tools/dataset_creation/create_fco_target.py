import os
import ast
import importlib
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.getcwd())
from utils.bev_utils import get_intersection_image_from_vehicleids

"""
This script pre-processes the original dataset to generate FCO (Future Collision Object) targets based on a specified sequence length.
Pre-processing the dataset in this manner allows for faster training of spatio-temporal networks by creating the FCO targets in advance.
To ensure efficient training with a specific sequence length, run this pre-processing script beforehand to generate the corresponding FCO targets.
"""

def create_fco_target(dataset_path: str, sequence_len: int):
    # load the csv dataset file into a pandas dataframe
    df = pd.read_csv(os.path.join(dataset_path, 'dataset.csv'))
    df['complete_vehicle_infos'] = df['complete_vehicle_infos'].apply(ast.literal_eval)
    df['fco_vehicle_infos'] = df['fco_vehicle_infos'].apply(ast.literal_eval)
    df['detected_vehicle_infos'] = df['detected_vehicle_infos'].apply(ast.literal_eval)

    # load the intersection data
    intersections = pd.read_csv(os.path.join(dataset_path, 'intersections.csv'))
    # convert the df to dict
    intersections_dict = intersections.set_index('id').T.to_dict()


    # import the config from dataset_path.configs.config.FCO import MULTI_FCO_INTERSECTION
    #list dirs under dataset_path.configs.config
    configs = os.listdir(os.path.join(dataset_path, 'configs'))
    config_path = os.path.join(dataset_path, 'configs', configs[0])
    config = importlib.import_module(config_path.replace('/', '.').replace('.py', ''))
    config = getattr(config, 'MULTI_FCO_INTERSECTION')

    min_time = int(df['time'].min())

    fco_time_diff = list()
    fco_complete_diff = list()
    complete = list()
    current = list()
    fco_time = list()
    assigned_times = list() 

    # create new dir for the created plots 
    plot_dir = os.path.join(dataset_path, f'fco_time_{sequence_len}')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.join(plot_dir, 'Analysis'), exist_ok=True)

    # create new column in the dataframe for the fco_time_dict
    df[f'fco_time_dict_{sequence_len}'] = None

    # iterate over the dataframe
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f'Create FCO target for {dataset_path} and sequence_len {sequence_len}'):
        if int(row.time) - min_time <= sequence_len:
            continue
        # init a list to track the vehicles
        fco_time_list = list()
        current_fco = list(set(list(df.loc[index]['fco_vehicle_infos'].keys()) + list(df.loc[index]['detected_vehicle_infos'].keys())))
        for i in range(sequence_len+1):
            new_time = row.time - i
            # find the index of the row with the new time
            time_indexes = df[df['time'] == new_time].index
            if len(time_indexes) == 0:
                raise ValueError(f'No index found for time {new_time}')
            elif len(time_indexes) == 1:
                # get the id of the row with the new time
                current_id = df[df['time'] == new_time]['id'].values[0]
            else:
                current_id = row.id.replace(f'_{int(row.id.split("_")[-2])}_', f'_{new_time}_')
            try:
                fco_time_list.extend(df.loc[df.id == current_id]['fco_vehicle_infos'].values[0].keys())
                fco_time_list.extend(df.loc[df.id == current_id]['detected_vehicle_infos'].values[0].keys())
            except:
                print(f'index {current_id} not found')
                continue
            
        # build set of unique vehicles
        fco_time_list = list(set(fco_time_list))

        # delete vehicles that are not in the current complete vehicle infos
        fco_time_list = [veh for veh in fco_time_list if veh in df.loc[index]['complete_vehicle_infos']]

        # create dict with the vehicles and their positions
        fco_time_dict = dict()
        for veh in fco_time_list:
            fco_time_dict[veh] = df.loc[index]['complete_vehicle_infos'][veh]
        
        # add the fco_time_dict to the dataframe
        df.at[index, f'fco_time_dict_{sequence_len}'] = fco_time_dict

        # get the intersection information
        # iterate through the intersections
        for _, intersection in intersections.iterrows():
            # check if the intersection string is part if the current id
            if intersection.id in row.id:
                current_intersection = intersection.id
        
        fig = get_intersection_image_from_vehicleids(current_intersection, intersections_dict, fco_time_dict, config['RADIUS'])
        plt.draw()
        fig.savefig(os.path.join(plot_dir, f'{index}.png'), bbox_inches='tight', pad_inches=0)
        plt.close(fig)


        fco_time_diff.append(len(fco_time_list)-len(current_fco)) # the vehicles can be see in the last sequence_len frames but not in the current frame
        fco_complete_diff.append(len(df.loc[index]["complete_vehicle_infos"])-len(fco_time_list)) # the vehicles that are in the current frame but not in the last sequence_len frames
        complete.append(len(df.loc[index]["complete_vehicle_infos"])) # the vehicles are in the current frame
        current.append(len(current_fco)) # the vehicles are in the current frame
        fco_time.append(len(fco_time_list))
        assigned_times.append(df.loc[index]['time'])
    
    # save the lists
    np.save(os.path.join(plot_dir, 'Analysis', 'fco_time_diff.npy'), fco_time_diff)
    np.save(os.path.join(plot_dir, 'Analysis', 'fco_complete_diff.npy'), fco_complete_diff)
    np.save(os.path.join(plot_dir, 'Analysis', 'complete.npy'), complete)
    np.save(os.path.join(plot_dir, 'Analysis', 'fco_current.npy'), current)
    np.save(os.path.join(plot_dir, 'Analysis', 'fco_time.npy'), fco_time)
    np.save(os.path.join(plot_dir, 'Analysis', 'assigned_times.npy'), assigned_times)

    print(f'mean fco_time_diff: {np.mean(fco_time_diff)}')
    print(f'mean fco_complete_diff: {np.mean(fco_complete_diff)}')
    print(f'mean complete: {np.mean(complete)}')
    print(f'mean current: {np.mean(current)}')

    # save the dataframe with the new column containing the fco_time_dict information for the current sequence_length without the first row
    df.to_csv(os.path.join(dataset_path, 'dataset.csv'))


if __name__ == "__main__":
    path = "data/i3040_newdetector_10p_cv_bigger"
    sequence_len = 5
    from configs.config import special_temporal_decoder_new as cfg
            # create the dataset target according to the current dataset if it does not exist
    if not os.path.exists(os.path.join(path, f'fco_time_{sequence_len}')):
        create_fco_target(path, sequence_len)
    else:
        print(f'fco target already exists')
    print('done')