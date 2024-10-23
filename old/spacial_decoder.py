import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
import PIL
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from old.train_utils_old import merge_sequence
from sub_models.decoder import Autoencoder
from utils.autoencoder_utils import show_image, get_current_output, create_train_gif, MyDataset, show_input_output, prepare_datasets, prepare_path_structure
#from utils.utils import prepare_datasets, prepare_path_structure
import wandb
import time
from datetime import datetime
import os
from configs.config import special_decoder as cfg
import shutil
from models import SpacialDecoder
from typing import Dict, List
from configs.configs_seqdataset import SEQ_DATASET
from utils.eval_utils import analyze_results
from old.test_std import test, test_dropout, test_prediction
from utils.train_utils import LRSchedulerTransformer, LargeDataset, unmerge_sequence, apply_image_dropout
from utils.logger_utils import logger
import logging

def plot_input_output(input_tensor: torch.Tensor, output_tensor: torch.Tensor, path: str, epoch: int):
    # convert tensor to image
    fig, axs = plt.subplots(2, 1)
    # add title to the axes
    axs[0].set_title('Input')
    axs[0].axis('off')
    axs[0].imshow(input_tensor.permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
    axs[1].set_title('Output')
    axs[1].axis('off')
    axs[1].imshow(output_tensor.permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
    plt.savefig(os.path.join(path, 'plots', f'input_output_{epoch}.png'))
    plt.close(fig)

@logger
def train_spacial_decoder(model: torch.nn.Module, train_IDs: List[str], val_IDs: List[str],
                                    device: str, path: str, num_epochs: int, scheduler) -> torch.nn.Module:
    # Define the loss function and optimization algorithm
    criterion = nn.BCELoss()
    # criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=scheduler['init_lr'])
    if scheduler['type'] == 'default':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma'])
    elif scheduler['type'] == 'transformer':
        scheduler = LRSchedulerTransformer(1024, warmup_steps=50)
    else:
        raise NotImplementedError
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_samples = 0
        model.train()
        for ID in train_IDs:
            logger.info(f'training on {ID} for epoch {epoch}')
            train_set = LargeDataset(ID)
            train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=8)
            logger.info('starting training')
            for input_image in train_loader:
                    total_samples += input_image.size(0)
                    input_image = input_image.to(device)
                    # Forward pass
                    outputs = model(input_image)
                    loss = criterion(outputs, input_image)
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    plot_input_output(input_image[0], outputs[0], path, epoch)
        scheduler.step()
        average_running_loss = running_loss / total_samples
        wandb.log({'loss': average_running_loss, 'lr': scheduler.get_last_lr()[0], 'epoch': epoch})
        logger.info(f"Epoch [{epoch}/{num_epochs}], Loss: {average_running_loss:.6f}")

        # validation
        if epoch % cfg['validation_frequency'] == 0:
            val_loss = 0.0
            num_val_samples = 0
            model.eval()  # set model to eval mode
            for ID in val_IDs:
                val_set = LargeDataset(ID)
                val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=8)
                with torch.no_grad():
                    for input_image in val_loader:
                
                        num_val_samples += input_image.size(0)
                        input_image = input_image.to(device)
                        outputs = model(input_image)
                        loss = criterion(outputs, input_image)
                        val_loss += loss.item()
            average_val_loss = val_loss / num_val_samples
            logger.info(f"Validation Loss after {epoch} epochs: {average_val_loss:.6f}")
            wandb.log({'val_loss': average_val_loss, 'epoch': epoch})
            if val_loss < best_val_loss:
                logger.info(f'new best model with val loss {val_loss} at epoch {epoch}')
                torch.save(model.state_dict(), os.path.join(path, 'models', 'best_val_statedict.pth'))
    torch.save(model.state_dict(), os.path.join(path, 'models', 'final_statedict.pth'))

    return model

if __name__ == '__main__':
    # prepare the run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # define the dataset
    train_IDs, val_IDs, test_IDs = prepare_datasets(cfg['dataset'], None, None, cfg['train_ids'],
                                                    cfg['val_ids'], cfg['test_ids'])
    
    # create the path structure
    filename = f'{cfg["dataset"]}_ {datetime.now().strftime("%d-%m_%H-%M-%S")}'
    path = prepare_path_structure(filename, 'spacial_decoder')

    # initialize wandb
    wandb.init(project="solving_occlusion_spacial_decoder", mode="online")
    wandb.config.update(cfg)

    # create the model
    logging.info('creating model')
    model = SpacialDecoder().to(device)
    
    # train and validate the model during training
    logging.info('starting training')
    model = train_spacial_decoder(model, train_IDs, val_IDs,
                                    device, path, cfg['num_epochs'], cfg['scheduler'])
    logging.info(f'finished training')



    
