import numpy as np
import pandas as pd
import ast
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import os
from torchvision.transforms import ToTensor, Resize, Compose
from tools.analyzation.plot_single_tensor import plot_single_tensor


class TemporalDataset(Dataset):
    def __init__(self, path: str, sequence_len: int, image_size: int, occluded_only: bool, map_binary: bool):
        self.sequence_len = sequence_len
        self.image_size = image_size
        self.path = path
        self.map_binary = map_binary
        # change dataset to id, time, intersection_info, complete_vehicle_infos, fco_vehicle_infos, detected_vehicle_infos (including fco), detected_{time}_vehicle_infos
        self.dataset = pd.read_csv(os.path.join(path, 'dataset.csv')).reset_index(drop=True)
        self.dataset['intersection'] = self.dataset['intersection'].apply(ast.literal_eval)
        self.dataset['complete_vehicle_infos'] = self.dataset['complete_vehicle_infos'].apply(ast.literal_eval)
        self.dataset['fco_vehicle_infos'] = self.dataset['fco_vehicle_infos'].apply(ast.literal_eval)
        self.dataset['detected_vehicle_infos'] = self.dataset['detected_vehicle_infos'].apply(ast.literal_eval)
        # remove all rows where there are nan values in the detected_time_{sequence_len}_vehicle_infos
        self.dataset = self.dataset[self.dataset[f'fco_time_dict_{sequence_len}'].notna()].reset_index(drop=True)
        self.dataset['detected_time_dict'] = self.dataset[f'fco_time_dict_{sequence_len}'].apply(ast.literal_eval)
        self.dataset['intersection_id'] = self.dataset['intersection'].apply(lambda x: list(x.keys())[0])

        self.fco_mintime_dict = dict.fromkeys(self.dataset['intersection_id'].unique().tolist())
        for key in self.fco_mintime_dict.keys():
            self.fco_mintime_dict[key] = self.dataset[self.dataset['intersection_id'] == key]['time'].min()
        

        if occluded_only:
            self.dataset['len_detected'] = self.dataset['detected_vehicle_infos'].apply(lambda x: len(x.keys()))
            self.dataset['len_detected_time'] = self.dataset[f'detected_time_{sequence_len}_vehicle_infos'].apply(lambda x: len(x.keys()))
            self.dataset['potential_occluded'] = self.dataset['len_detected'] - self.dataset['len_detected_time']
            # create new dataset that only contains the occluded vehicles
            self.index_dataset = self.dataset[self.dataset['potential_occluded'] > 0].index
        else:
            self.index_dataset = self.dataset


        self.transforms = Compose([
            Resize((self.image_size, self.image_size)),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.index_dataset)

    def __getitem__(self, idx):
        # map the idx to the index of the dataset
        current_id = self.index_dataset.iloc[idx]['id']
        idx = self.dataset[self.dataset['id'] == current_id].index[0]
        current_time = self.dataset.loc[idx, 'time']
        if current_time - self.sequence_len < self.fco_mintime_dict[self.dataset.loc[idx, 'intersection_id']]:
            current_time = current_time + 2 * self.sequence_len
        # get the timestamps for teh sequence
        sequence_times = torch.arange(current_time, current_time - (self.sequence_len) - 1, -1)
        # reverse the sequence times --> oldest entry is first [0]
        sequence_times = sequence_times.flip(0)
        # get the sequence names from the sequence times
        if len(self.dataset[self.dataset['time'] == sequence_times.tolist()[0]]['id']) == 1:
            sequence_names = [self.dataset[self.dataset['time'] == sequence_times.tolist()[0]]['id'].values[0] for i in sequence_times.tolist()]
        else:
            t = "_" + current_id.split('_')[-2] + "_"
            sequence_names = [f'{current_id.replace(t, "_" + str(i) + "_")}' for i in sequence_times.tolist()]
        # load the images
        try:
            complete_bev_images = torch.stack([self.transforms(Image.open(os.path.join(self.path, 'complete_bev_images', f'{i}.png')).convert('L')) for i in sequence_names])
            detected_bev_images = torch.stack([self.transforms(Image.open(os.path.join(self.path, 'detected_bev_images', f'{i}.png')).convert('L')) for i in sequence_names])
            detected_time_bev_images = torch.stack([self.transforms(Image.open(os.path.join(self.path, f'fco_time_{self.sequence_len}', f'{i}.png')).convert('L')) for i in sequence_names])
        except:
            print('error')

        # get the current vehicle dicts
        complete_vehicle_dict = self.dataset.loc[idx, 'complete_vehicle_infos']
        detected_vehicle_dict = self.dataset.loc[idx, 'detected_vehicle_infos']
        detected_time_vehicle_dict = self.dataset.loc[idx, 'detected_time_dict']

        if self.map_binary:
            # map the complete, detected and detected_time images to binary images with threshold 0.5
            complete_bev_images = (complete_bev_images > 0.5).float()
            detected_bev_images = (detected_bev_images > 0.25).float()
            detected_time_bev_images = (detected_time_bev_images > 0.5).float()

        return {'complete_bev_images': complete_bev_images, 'detected_bev_images': detected_bev_images, 'detected_time_bev_images': detected_time_bev_images, 'sequence_times': sequence_times, 'complete_vehicle_dict': complete_vehicle_dict, 'detected_vehicle_dict': detected_vehicle_dict, 'detected_time_vehicle_dict': detected_time_vehicle_dict}


class PretrainingDataset(Dataset):
    def __init__(self, path: str, sequence_len: int, image_size: int):
        self.sequence_len = sequence_len
        self.image_size = image_size
        self.path = path
        # change dataset to id, time, intersection_info, complete_vehicle_infos, fco_vehicle_infos, detected_vehicle_infos (including fco), detected_{time}_vehicle_infos
        self.dataset = pd.read_csv(os.path.join(path, 'dataset.csv')).rename(columns={'Unnamed: 0': 'id'}).reset_index(drop=True)
        self.dataset['intersection'] = self.dataset['intersection'].apply(ast.literal_eval)
        self.dataset['complete_vehicle_infos'] = self.dataset['complete_vehicle_infos'].apply(ast.literal_eval)
        self.dataset['fco_vehicle_infos'] = self.dataset['fco_vehicle_infos'].apply(ast.literal_eval)
        self.dataset['detected_vehicle_infos'] = self.dataset['detected_vehicle_infos'].apply(ast.literal_eval)
        self.dataset[f'detected_time_{sequence_len}_vehicle_infos'] = self.dataset[f'detected_time_{sequence_len}_vehicle_infos'].apply(ast.literal_eval)
        self.dataset['intersection_id'] = self.dataset['intersection'].apply(lambda x: list(x.keys())[0])
        self.fco_mintime_dict = dict.fromkeys(self.dataset['intersection_id'].unique().tolist())
        for key in self.fco_mintime_dict.keys():
            self.fco_mintime_dict[key] = self.dataset[self.dataset['intersection_id'] == key]['time'].min()


def collate_fn(batch):
    # batch the dict items
    batched_data = {key: [] for key in batch[0].keys()}
    for item in batch:
        for key in batched_data.keys():
            batched_data[key].append(item[key])
    
    for key in ['complete_bev_images', 'detected_bev_images', 'detected_time_bev_images']:
        batched_data[key] = torch.stack(batched_data[key])
    
    return batched_data
    


def get_dataset(cfg, transform):
    #pre_train = cfg['pre_train']
    # if pre_train:
    #     if cfg['pre_train_path'] is not None:
    #         dataset_path = os.path.join('pre_train_data', cfg['pre_train_path'])
    #         dataset = PretrainDataset(dataset_path, transform, org_data = False)
    #     else:
    #         dataset = PretrainDataset(os.path.join(cfg['dataset_path'], cfg['dataset_name'], 'bev_images'), transform, org_data = True)
    # else:
    dataset = TemporalDataset(os.path.join(cfg['dataset_path'], cfg['dataset_name']), cfg['sequence_len'], cfg['image_size'])
    return dataset


if __name__ == "__main__":
    pass