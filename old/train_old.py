import logging
import math
import os
import pickle
import shutil
import time
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

from configs.config import network_configs, special_temporal_decoder_new as cfg
from configs.configs_seqdataset import SEQ_DATASET
from models import SpacialTemporalDecoder
from utils.autoencoder_utils import prepare_path_structure
from utils.criterion_utils import get_criterion, CombinedLoss
from utils.dataset_utils import TemporalDataset, get_dataset, collate_fn
from utils.eval_utils import analyze_results, get_mean_iou
from utils.logger_utils import logger
from utils.scheduler_utils import create_scheduler
from old.test_std import test, test_dropout, test_prediction
from utils.train_utils import apply_image_dropout, chunked_indices_split, set_seed, save_model, log_metrics, transformations
from utils.transformations_utils import get_transformations
from utils.wandb_utils import start_wandb


from create_fco_target import create_fco_target

print(os.getcwd())
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def process_batch(batch, model, criterion, device, pre_train):
    complete_bev_images = batch['complete_bev_images']
    detected_bev_images = batch['detected_bev_images']
    detected_time_bev_images = batch['detected_time_bev_images']
    sequence_times = batch['sequence_times']
    complete_vehicle_dict = batch['complete_vehicle_dict']
    detected_vehicle_dict = batch['detected_vehicle_dict']
    detected_time_vehicle_dict = batch['detected_time_vehicle_dict']
    target_sequence = detected_bev_images if pre_train else detected_time_bev_images
    target = target_sequence[:, -1, ...]
    input_sequence = detected_bev_images[:, -1, ...].unsqueeze(dim=1) if pre_train else detected_bev_images
    input_sequence = input_sequence.to(device) if pre_train else input_sequence.permute(1, 0, 2, 3, 4).to(device)
    target = target.to(device)
    outputs = model(input_sequence)
    loss, individual_losses = criterion(outputs, target)
    return loss, individual_losses, outputs, target, input_sequence, target_sequence


def train_epoch(model, train_loader, optimizer, criterion, scheduler, device, pre_train, epoch, path):
    model.train()
    running_loss, running_iou, total_samples = 0.0, 0.0, 0
    running_individual_losses = dict.fromkeys(criterion.losses.keys(), 0.0)
    for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
        loss, individual_losses, outputs, biv_target, input_sequence, target_sequence = process_batch(batch, model, criterion, device, pre_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_iou += get_mean_iou(outputs, biv_target)
        total_samples += 1
        for key, value in individual_losses.items():
            running_individual_losses[key] += value
    average_running_loss = running_loss / total_samples
    mean_iou = running_iou / total_samples
    for key in running_individual_losses:
        running_individual_losses[key] /= total_samples
    analyze_results(outputs, target_sequence, input_sequence, time, 0, os.path.join(path, 'plots', 'train'), epoch, save_plot = True,
        overlay_image_path = None)

    return average_running_loss, mean_iou, running_individual_losses


def validate_epoch(model, val_loader, criterion, device, cfg, pre_train, epoch, path):
    model.eval()
    running_loss, running_iou, total_samples = 0.0, 0.0, 0
    running_individual_losses = dict.fromkeys(criterion.losses.keys(), 0.0)
    with torch.no_grad():
        for batch in val_loader:
            loss, individual_losses, outputs, biv_target, input_sequence, target_sequence = process_batch(batch, model, criterion, device, pre_train)
            running_loss += loss.item()
            running_iou += get_mean_iou(outputs, biv_target)
            total_samples += 1
            for key, value in individual_losses.items():
                running_individual_losses[key] += value
    average_running_loss = running_loss / total_samples
    mean_iou = running_iou / total_samples
    for key in running_individual_losses:
        running_individual_losses[key] /= total_samples
    analyze_results(outputs, target_sequence, input_sequence, time, 0, os.path.join(path, 'plots', 'val'), epoch, save_plot = True,
            overlay_image_path = None)

    return average_running_loss, mean_iou, running_individual_losses

@logger
def train(cfg, model, train_loader, val_loader, device, path, pre_train=False):
    # Define the loss function and optimization algorithm
    criterion = CombinedLoss(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg['scheduler']['init_lr'])
    scheduler = create_scheduler(optimizer)

    best_val_loss = float('inf')
    for epoch in range(cfg['num_epochs']):
        train_loss, train_iou, train_individual_losses = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, pre_train, epoch, path
        )
        log_metrics('train', epoch, train_loss, train_iou, train_individual_losses)
        

        if epoch % cfg['validation_frequency'] == 0:
            val_loss, val_iou, val_individual_losses = validate_epoch(
                model, val_loader, criterion, device, cfg, pre_train, epoch, path
            )
            log_metrics('val', epoch, val_loss, val_iou, val_individual_losses)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, path, epoch, cfg)


if __name__ == '__main__':
    # prepare the run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # set the seed
    set_seed(42)

    # create the dataset target according to the current dataset if it does not exist
    if not os.path.exists(f'data/{cfg["dataset_name_train"]}/detected_vehicles_time_{cfg["sequence_len"]}'):
        create_fco_target(f'data/{cfg["dataset_name_train"]}', cfg['sequence_len'])
        create_fco_target(f'data/{cfg["dataset_name_val"]}', cfg['sequence_len'])
    else:
        print(f'fco target already exists')
    
    train_dataset = TemporalDataset(os.path.join(cfg['dataset_path'], cfg["dataset_name_train"]), cfg['sequence_len'], cfg['image_size'])
    val_dataset = TemporalDataset(os.path.join(cfg['dataset_path'], cfg["dataset_name_val"]), cfg['sequence_len'], cfg['image_size'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=8, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=8, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    
    # create the path structure
    filename = f'{cfg["dataset_name_train"]}_{datetime.now().strftime("%d-%m_%H-%M-%S")}'
    path = prepare_path_structure(filename, 'trained_spacial_temporal_decoder')

    # initialize wandb
    start_wandb(cfg, filename)

    # create the model
    logging.info('creating model')
    model = SpacialTemporalDecoder(cfg, network_configs).to(device)
    if cfg['load_complete_model'] is not None:
        model.load_state_dict(torch.load(cfg['load_complete_model']), strict=False)

    logging.info('start the training')
    model = train(cfg, model, train_loader, val_loader, device, path, pre_train = cfg['pre_train'])
    logging.info(f'finished training')


    
