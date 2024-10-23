import pandas as pd
import numpy as np
import ast
import os


if __name__ == "__main__":
    path = "data/i3040_newdetector_10p_cv_bigger"

    # load the dataset
    dataset = pd.read_csv(os.path.join(path, 'dataset.csv'))
    print(dataset.keys())

    # remove rows where the fco_time_dict is nan
    dataset = dataset[~pd.isna(dataset['fco_time_dict_5'])] 

    dataset['detected_vehicle_infos'] = dataset['detected_vehicle_infos'].apply(ast.literal_eval)
    dataset['fco_time_dict_5'] = dataset['fco_time_dict_5'].apply(ast.literal_eval)

    dataset['len_detected'] = dataset['detected_vehicle_infos'].apply(lambda x: len(x))
    dataset['len_fco'] = dataset['fco_time_dict_5'].apply(lambda x: len(x))

    dataset['temporal_potential'] = dataset['len_fco'] - dataset['len_detected']

    # print the rows where the temporal potential is bigger than 5
    print(dataset[dataset['temporal_potential'] > 3])
