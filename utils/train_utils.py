from typing import Dict, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os
from configs.configs_seqdataset import SEQ_DATASET
import PIL
import time
from torchvision import transforms
import wandb
import time
import pandas as pd
import random


def unmerge_sequence(sequence: torch.Tensor, base_key: str) -> Dict[str, np.array]:
    '''
    function takes the merged sequence tensor and unmerges it into a dictionary with correct temporal keys
    '''
    sequence_len = sequence.shape[0]
    keys = [base_key]
    for i in range(1, sequence_len):
        key = f'{base_key}-{i}'
        keys.append(key)
    keys.reverse()
    sequence_dict = {}
    for i, key in enumerate(keys):
        sequence_dict[key] = sequence[i, ...].detach().cpu().numpy()
    
    return sequence_dict

def apply_image_dropout(input_sequence: torch.Tensor, dropout_prob: float) -> torch.Tensor:
    sequence = input_sequence.permute(1, 0, 2, 3, 4)
    zero_tensor = torch.zeros_like(sequence[0][0]).to(sequence.device)
    for b, batch_item in enumerate(sequence):
        for i, image in enumerate(batch_item):
            if torch.rand(1) < dropout_prob:
                sequence[b][i] = zero_tensor
    sequence = sequence.permute(1, 0, 2, 3, 4)
    return sequence

def chunked_indices_split(total_length: int, chunk_size: int, train_ratio: float=0.8, only_nonseen: bool=False, dataset_path: Union[str, None]=None, sequence_len: int=1):
    assert train_ratio <= 1.0 and train_ratio >= 0.0 # train_ratio must be between 0 and 1
    # Create index chunks
    indices = list(range(total_length))
    index_chunks = [indices[i:i + chunk_size] for i in range(0, total_length, chunk_size)]
    
    # Shuffle index chunks
    random.shuffle(index_chunks)
    
    # Split index chunks into train and validation sets
    num_train_chunks = int(len(index_chunks) * train_ratio)
    train_idx_chunks = index_chunks[:num_train_chunks]
    val_idx_chunks = index_chunks[num_train_chunks:]
    
    # Flatten index chunks back into lists
    train_idx = [idx for chunk in train_idx_chunks for idx in chunk]
    val_idx = [idx for chunk in val_idx_chunks for idx in chunk]

    if only_nonseen:
        train_idx = get_nonseen_idx(train_idx, dataset_path, sequence_len)
        val_idx = get_nonseen_idx(val_idx, dataset_path, sequence_len)
    return train_idx, val_idx

def get_nonseen_idx(idx_list, path, sequence_len):
    # open the dataset
    dataset = pd.read_csv(os.path.join(path, 'dataset.csv'))
    nonseen_idx_list = []
    for idx in idx_list:
       # get the 'diff' column of the current idx
        diff = dataset.iloc[idx][f'diff_{sequence_len}']
        if diff > 0:
            nonseen_idx_list.append(idx)
    return nonseen_idx_list


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def save_model(model, path, epoch, cfg):
    model_save_path = os.path.join(path, 'models', f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_save_path)
    #logger.info(f'Model saved at {model_save_path}')


def log_metrics(phase, epoch, loss, iou, individual_losses):
    metrics = {
        f'{phase}_loss': loss,
        f'{phase}_iou': iou,
        **{f'{phase}_{key}_loss': value for key, value in individual_losses.items()}
    }
    metrics['epoch'] = epoch
    wandb.log(metrics)

transformations = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor()
])
