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
from sub_models.old_models import SpacialTemporalDecoderOld
from utils.autoencoder_utils import show_image, get_current_output, create_train_gif, MyDataset, show_input_output, prepare_datasets, prepare_path_structure
#from utils.utils import prepare_datasets, prepare_path_structure
import wandb
import time
from datetime import datetime
import os
from configs.config import special_temporal_decoder as cfg
import shutil
from models import SpacialTemporalDecoder
from typing import Dict, List
from configs.configs_seqdataset import SEQ_DATASET
from utils.eval_utils import analyze_results
print(os.getcwd())
from old.test_std import test, test_dropout, test_prediction
from utils.train_utils import LRSchedulerTransformer, LargeSequentialDataset, unmerge_sequence, apply_image_dropout
from utils.logger_utils import logger
import logging
import tqdm

@logger
def train_spacial_temporal_decoder(model: torch.nn.Module, train_IDs: List[str], val_IDs: List[str],
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
            train_set = LargeSequentialDataset(ID)
            print(cfg['batch_size'])
            train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=8)
            logger.info('starting training')
            for input_sequence, target_sequence, time in train_loader:
                    cfg['image_dropout'] = 0.1
                    input_sequence = input_sequence.squeeze(dim=2)
                    target_sequence = target_sequence.squeeze(dim=2)
                    #print(f'input sequence shape: {input_sequence.shape}')
                    input_sequence = apply_image_dropout(input_sequence, cfg['image_dropout'])
                    biv_target = target_sequence[:, -1, ...]
                    total_samples += input_sequence.size(0)
                    # Permute the input sequence to fit the transformer input
                    input_sequence = input_sequence.permute(1, 0, 2, 3, 4)
                    # Get a batch of inputs
                    input_sequence, biv_target = input_sequence.to(device), biv_target.to(device)
                    # Forward pass
                    outputs = model(input_sequence)
                    loss = criterion(outputs, biv_target)
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    scheduler.step()
        average_running_loss = running_loss / total_samples
        wandb.log({'loss': average_running_loss, 'lr': scheduler.get_last_lr()[0], 'epoch': epoch})
        logger.info(f"Epoch [{epoch}/{num_epochs}], Loss: {average_running_loss:.6f}")
        analyze_results(outputs, target_sequence, input_sequence, time, 0, os.path.join(path, 'plots', 'train'), epoch, train_set.dataset, save_plot = True,
        overlay_image_path = "content/background_3040.png")

        # validation
        if epoch % cfg['validation_frequency'] == 0:
            val_loss = 0.0
            num_val_samples = 0
            model.eval()  # set model to eval mode
            for ID in val_IDs:
                val_set = LargeSequentialDataset(ID)
                val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=8)
                with torch.no_grad():
                    for input_sequence, target_sequence, time in val_loader:
                        input_sequence = input_sequence.squeeze(dim=2)
                        target_sequence = target_sequence.squeeze(dim=2)
                        biv_target = target_sequence[:, -1, ...]
                        num_val_samples += input_sequence.size(0)
                        input_sequence = input_sequence.permute(1, 0, 2, 3, 4)
                        input_sequence, biv_target = input_sequence.to(device), biv_target.to(device)
                        outputs = model(input_sequence)
                        loss = criterion(outputs, biv_target)
                        val_loss += loss.item()
            average_val_loss = val_loss / num_val_samples
            logger.info(f"Validation Loss after {epoch} epochs: {average_val_loss:.6f}")
            analyze_results(outputs, target_sequence, input_sequence, time, 0, os.path.join(path, 'plots', 'val'), epoch, val_set.dataset, save_plot = True,
            overlay_image_path = 'content/background_3040.png')
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
    train_IDs, val_IDs, test_IDs = prepare_datasets(cfg['dataset_path'], cfg['dataset_name'], cfg['sequence_len'], cfg['train_ids'],
                                                    cfg['val_ids'], cfg['test_ids'])
    
    # create the path structure
    filename = f'{cfg["dataset_name"]}_{datetime.now().strftime("%d-%m_%H-%M-%S")}'
    path = prepare_path_structure(filename, 'trained_spacial_temporal_decoder')

    # initialize wandb
    # check if program is run in debug mode
    # wandb_mode = 'disabled' if __debug__ else 'online'
    wandb_mode = 'online'
    print(f'wandb mode: {wandb_mode}')
    wandb.init(project="solving_occlusion_spacial_temporal_decoder", mode=wandb_mode)
    wandb.config.update(cfg)

    # create the model
    logging.info('creating model')
    model = SpacialTemporalDecoder(
        cfg,
        fix_decoder=cfg['fix_decoder'], load_decoder=cfg['load_decoder'],
        fix_spacial=cfg['fix_spacial'], load_spacial=cfg['load_spacial']
    ).to(device)
    if cfg['load_path'] is not None:
        logging.info(f'loading model from {cfg["load_path"]}')
        model.load_state_dict(torch.load(cfg['load_path']))
    else:
        logging.info(f'no model loaded')
    
    # train and validate the model during training
    logging.info('starting training')
    model = train_spacial_temporal_decoder(model, train_IDs, val_IDs,
                                    device, path, cfg['num_epochs'], cfg['scheduler'])
    logging.info(f'finished training')

    # test the model
    logging.info(f'testing model')
    test_prediction(model, test_IDs, device, path, cfg, cfg['test_prediction_steps'])

    # archive the results
    shutil.make_archive(os.path.join('archive', filename), 'zip', os.path.join(path, 'plots'))
    time.sleep(5)
    logging.info(f'finished zipping')
    wandb.save(f'archive/{filename}.zip')
    # wandb.save(os.path.join(path, 'models', 'full'))
    logging.info(f'finished archiving')



    
