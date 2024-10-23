import torch
import numpy as np
import pickle
from old.train_utils_old import merge_sequence
import pandas as pd
from configs.configs_seqdataset import SEQ_DATASET
import os


def merge_sequential_merge(ID: str):
    with open(ID, "rb") as file:
        dataset = pickle.load(file)
    # create a dictionary with the input sequence
    input_sequence = merge_sequence(dataset, 'biv_t', SEQ_DATASET['SEQU_LEN'])
    input_sequence = pd.DataFrame.from_dict(input_sequence, orient='index')
    transposed_sequence = input_sequence.T
    input_sequence = {i: value for i, (_, value) in enumerate(transposed_sequence.items())}


    # create a dictionary with the target sequence
    target_sequence = merge_sequence(dataset, 'biv_target_t', SEQ_DATASET['SEQU_LEN'])
    target_sequence = pd.DataFrame.from_dict(target_sequence, orient='index')
    transposed_sequence = target_sequence.T
    target_sequence = {i: value for i, (_, value) in enumerate(transposed_sequence.items())}

    # create a dictionary with the detected vehicles
    detected_vehicles = merge_sequence(dataset, 'detected_vehicles', SEQ_DATASET['SEQU_LEN'], tens=False)
    detected_vehicles = pd.DataFrame.from_dict(detected_vehicles, orient='index')
    transposed_sequence = detected_vehicles.T
    detected_vehicles = {i: value for i, (_, value) in enumerate(transposed_sequence.items())}

    # create a dictionary with the times of the images
    times = merge_sequence(dataset, 'time', SEQ_DATASET['SEQU_LEN'], tens=False)
    times = pd.DataFrame.from_dict(times, orient='index')
    transposed_sequence = times.T

    # create a dictionary with all vehicles
    all_vehicles = merge_sequence(dataset, 'all_vehicles', SEQ_DATASET['SEQU_LEN'], tens=False)
    all_vehicles = pd.DataFrame.from_dict(all_vehicles, orient='index')
    transposed_sequence = all_vehicles.T
    all_vehicles = {i: value for i, (_, value) in enumerate(transposed_sequence.items())}

    merged_dataset = {'input_sequence': input_sequence, 'target_sequence': target_sequence,
                        'detected_vehicles': detected_vehicles, 'times': times, 'all_vehicles': all_vehicles}
    
    # find the position to insert after the last / to insert the merged_ in the name
    last_slash = ID.rfind('/')
    new_ID = ID[:last_slash+1] + 'merged_' + ID[last_slash+1:]
    with open(new_ID, 'wb') as file:
        pickle.dump(merged_dataset, file)

if __name__ == "__main__":
    # path to the dataset
    path = 'data/i3040_newvector_am'
    sequence_len = 10

    merge_list= list()

    # add all files with the path and sequence_len to the merge_list
    # get all files in the path
    all_on_path = os.listdir(path)
    for file in all_on_path:
        if file.startswith(f'sequence{sequence_len}'):
            merge_list.append(path + '/' + file)

    for ID in merge_list:
        print(f'processing {ID}')
        merge_sequential_merge(ID)
