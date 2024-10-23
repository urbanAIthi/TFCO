import logging
import os
import time
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import network_configs, special_temporal_decoder_new as cfg
from models import SpacialTemporalDecoder
from utils.autoencoder_utils import prepare_path_structure
from utils.criterion_utils import CombinedLoss
from utils.dataset_utils import TemporalDataset, collate_fn
from utils.eval_utils import analyze_results, get_mean_iou
from utils.logger_utils import logger
from utils.scheduler_utils import create_scheduler
from utils.train_utils import set_seed, save_model, log_metrics
from utils.wandb_utils import start_wandb
from tools.dataset_creation.create_fco_target import create_fco_target

print(os.getcwd())
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def process_batch(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    criterion: CombinedLoss,
    device: torch.device,
    pre_train: bool
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Extract data from the batch
    complete_bev_images = batch['complete_bev_images']
    detected_bev_images = batch['detected_bev_images']
    detected_time_bev_images = batch['detected_time_bev_images']
    sequence_times = batch['sequence_times']
    complete_vehicle_dict = batch['complete_vehicle_dict']
    detected_vehicle_dict = batch['detected_vehicle_dict']
    detected_time_vehicle_dict = batch['detected_time_vehicle_dict']

    # Determine the target sequence based on whether we are in pre-training mode
    target_sequence = detected_bev_images if pre_train else detected_time_bev_images
    target = target_sequence[:, -1, ...]
    
    # Prepare the input sequence
    input_sequence = detected_bev_images[:, -1, ...].unsqueeze(dim=1) if pre_train else detected_bev_images
    input_sequence = input_sequence.to(device) if pre_train else input_sequence.permute(1, 0, 2, 3, 4).to(device)
    target_sequence = target_sequence.permute(1, 0, 2, 3, 4).to(device)
    target = target.to(device)

    # Forward pass through the model
    outputs = model(input_sequence)
    
    # Compute the loss
    loss, individual_losses = criterion(outputs, target)

    return loss, individual_losses, outputs, target, input_sequence, target_sequence


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: CombinedLoss,
    scheduler: lr_scheduler._LRScheduler,
    device: torch.device,
    pre_train: bool,
    epoch: int,
    path: str
    ) -> Tuple[float, float, Dict[str, float]]:
    model.train()
    running_loss, running_iou, total_samples = 0.0, 0.0, 0
    running_individual_losses = {key: 0.0 for key in criterion.losses.keys()}

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

    analyze_results(
        outputs[-1], target_sequence[-1], input_sequence[-1], 
        batch['sequence_times'][-1][-1].item(), epoch, 
        os.path.join(path, 'plots', 'train'), save_plot=True,
        overlay_image_path=None, pre_train=pre_train
    )

    return average_running_loss, mean_iou, running_individual_losses


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    cfg: Dict,
    pre_train: bool,
    epoch: int,
    path: str
    ) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    running_loss, running_iou, total_samples = 0.0, 0.0, 0
    running_individual_losses = {key: 0.0 for key in criterion.losses.keys()}
    
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

    analyze_results(
        outputs[-1], target_sequence[-1], input_sequence[-1], 
        batch['sequence_times'][-1][-1].item(), epoch, 
        os.path.join(path, 'plots', 'train'), save_plot=True,
        overlay_image_path=None, pre_train=pre_train
    )

    return average_running_loss, mean_iou, running_individual_losses

@logger
def train(cfg: Dict, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, path: str, pre_train: bool = False) -> None:
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
    # Prepare the run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Set the seed
    set_seed(42)

    # Create the path structure
    filename = f'{cfg["dataset_name_train"]}_{datetime.now().strftime("%d-%m_%H-%M-%S")}'
    path = prepare_path_structure(filename, 'trained_models', cfg['pre_train'], cfg['temporal'])

    # Create the dataset target if it does not exist
    train_dataset_path = os.path.join(cfg['dataset_path'], cfg['dataset_name_train'], f'fco_time_{cfg["sequence_len"]}')
    val_dataset_path = os.path.join(cfg['dataset_path'], cfg['dataset_name_val'], f'fco_time_{cfg["sequence_len"]}')
    if not os.path.exists(train_dataset_path):
        create_fco_target(os.path.join(cfg['dataset_path'], cfg['dataset_name_train']), cfg['sequence_len'])
        create_fco_target(os.path.join(cfg['dataset_path'], cfg['dataset_name_val']), cfg['sequence_len'])
    else:
        logger.info('FCO target already exists')

    # Create datasets and data loaders
    train_dataset = TemporalDataset(os.path.join(cfg['dataset_path'], cfg['dataset_name_train']), cfg['sequence_len'], cfg['image_size'], cfg['only_occluded'], cfg['map_binary'])
    val_dataset = TemporalDataset(os.path.join(cfg['dataset_path'], cfg['dataset_name_val']), cfg['sequence_len'], cfg['image_size'], cfg['only_occluded'], cfg['map_binary'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=8, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=8, shuffle=True, collate_fn=collate_fn, pin_memory=True)

    # Initialize Weights and Biases
    start_wandb(cfg, filename)

    # Create the model
    logger.info('Creating model')
    model = SpacialTemporalDecoder(cfg, network_configs).to(device)
    if cfg['load_complete_model']:
        model.load_state_dict(torch.load(cfg['load_complete_model']), strict=False)

    # Start the training
    logger.info('Starting the training')
    train(cfg, model, train_loader, val_loader, device, path, pre_train=cfg['pre_train'])
    logger.info('Finished training')


    
