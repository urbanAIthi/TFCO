import logging
import math
import os
import pickle
import shutil
import time
from datetime import datetime
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from configs.test_config import test_config
from configs.config import network_configs
from configs.configs_seqdataset import SEQ_DATASET
from models import SpacialTemporalDecoder
from utils.autoencoder_utils import prepare_path_structure
from utils.dataset_utils import NewDataset, collate_fn
from utils.eval_utils import analyze_results, get_mean_iou
from utils.logger_utils import logger
from old.test_std import test, test_dropout, test_prediction
from utils.train_utils import LRSchedulerTransformer, apply_image_dropout
from utils.transformations_utils import get_transformations
#from utils.scheduler_utils import get_scheduler
from utils.criterion_utils import get_criterion


def get_datapoints(dataset_path: str, dataset_name: str, test_samples: Union[str,List[str]],test_path) -> None:
    """
    Creates a modified dataset.csv with the selected datapoints
    """
    if test_samples == 'all':
        # copy the dataset csv to the test folder
        shutil.copy(os.path.join(dataset_path, dataset_name, 'dataset.csv'), os.path.join(test_path, 'dataset.csv'))
    
    elif type(test_samples) == list:
        # check that all values of the list are integers
        assert all(isinstance(x, int) for x in test_samples), "All values of the list must be integers"
        # load the initial dataset.csv
        dataset = pd.read_csv(os.path.join(dataset_path, dataset_name, 'dataset.csv'))
        dataset_times = dataset['time'].astype(str).str[1:-1].astype(int)
        # check that all values of test_samples are in the dataset
        assert all(x in dataset_times for x in test_samples), "All values of the list must be in the dataset"

        # pop all rows that are not in the test_samples list
        dataset = dataset[dataset['time'].isin(test_samples)]

        # save the modified dataset.csv to the test folder
        dataset.to_csv(os.path.join(test_path, 'dataset.csv'), index=False)

def prepare_test_folder(dataset_name, model_path):
    # create the test folder
    test_path = os.path.join('test_results', dataset_name, datetime.now().strftime("%d-%m_%H-%M-%S"))
    os.makedirs(test_path)
    
    # copy the model parameters to the test folder
    os.makedirs(os.path.join(test_path, 'model'))
    shutil.copy(model_path, os.path.join(test_path, 'model', 'model.pth'))

    # copy the configs to the test folder
    os.makedirs(os.path.join(test_path, 'configs'))
    shutil.copy('configs/test_config.py', os.path.join(test_path, 'configs', 'test_config.py'))
    # also copy the config from the model to the test folder
    shutil.copy(os.path.join(os.path.split(os.path.split(model_path)[0]), 'config.py'), os.path.join(test_path, 'configs', 'config.py'))

    #create the results
    os.makedirs(os.path.join(test_path, 'result_plots'))

if __name__ == "__main__":
    get_datapoints(test_config['dataset_path'], test_config['dataset_name'], test_config['test_samples'])
    prepare_test_folder(test_config['dataset_name'], test_config['model_path'])

    