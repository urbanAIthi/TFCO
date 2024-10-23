import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import Dataset, DataLoader
import PIL
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from sub_models.decoder import Autoencoder
from utils.autoencoder_utils import show_image, get_current_output, create_train_gif, MyDataset
import wandb
import time
from datetime import datetime
import os
from solving_occlusion.solving_occlusion.configs.config import autoencoder_config
import shutil
from sub_models.old_models import Trans_CNN_Autoencoder

def train_trans_cnn(autoencoder, dataloader, device, path):
    # Define the loss function and optimization algorithm
    criterion = nn.BCELoss()
    #criterion = CustomLoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=0.9, gamma=5)
    autoencoder = autoencoder.to(device)

    for epoch in range(100):
        running_loss = 0.0
        for batch in dataloader:
            # Get a batch of inputs
            batch = batch.to(device)

            # Forward pass
            outputs = autoencoder(batch)
            loss = criterion(outputs, batch)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Print the loss for this
        # Perform scheduler step after each epoch
        scheduler.step()
        wandb.log({'loss': running_loss, 'lr': scheduler.get_last_lr()[0]})
        print(f"Epoch [{epoch}/{100}], Loss: {running_loss:.6f}")
        if epoch % 10 == 0:
            trained_out = get_current_output(autoencoder, dataset, 0, device, forward_function=autoencoder.forward)
            show_image(trained_out, show=False, save_path=os.path.join(path, 'plots', f'train_epoch_{epoch}.png'))
    torch.save(autoencoder.state_dict(), os.path.join(path, 'models', 'trained_trans_cnn.pth'))


if __name__ == "__main__":
    filename = f"autoencoder_{datetime.now().strftime('%m-%d_%H-%M-%S')}"
    path = os.path.join('trained_trans_cnn', filename)
    if not os.path.exists('trained_trans_cnn'):
        os.mkdir('trained_trans_cnn')
    os.mkdir(path)
    os.mkdir(os.path.join(path, 'models'))
    os.mkdir(os.path.join(path, 'plots'))
    wandb.init(project="solving_occlusion_trans_cnn", mode="offline")
    with open("test.pkl", "rb") as f:
        data = pickle.load(f)
    data = {k: data[k] for k in list(data.keys())[100:]}
    data = pd.DataFrame.from_dict(data, orient='index')
    data = data.reset_index(drop=True)
    data = data.to_dict(orient='index')
    show_image(data[0][0], show=True, save_path=os.path.join(path, 'plots', 'target.png'))
    dataset = MyDataset(data)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    trans_cnn_autoencoder = Trans_CNN_Autoencoder().to(device)
    num_params = sum(p.numel() for p in trans_cnn_autoencoder.parameters())
    print("Number of parameters:", num_params)
    pre_out = get_current_output(trans_cnn_autoencoder, dataset, 0, device)
    print(f'Pre-training output shape: {pre_out.shape}')
    show_image(pre_out, show=False, save_path=os.path.join(path, 'plots', 'pre.png'))
    train_trans_cnn(trans_cnn_autoencoder, dataloader, device, path)
    trained_out = get_current_output(trans_cnn_autoencoder, dataset, 0, device)
    # show_input_output(dataset[1][0], trained_out)
    show_image(trained_out, show=False, save_path=os.path.join(path, 'plots', 'trained.png'))
    create_train_gif(path)
    print(f'finished training')
    shutil.make_archive(filename, 'zip', os.path.join(path, 'plots'))
    time.sleep(5)
    print(f'finished zipping')
    wandb.save(f'{filename}.zip')
    # wandb.save(os.path.join(path, 'models', 'full'))
    print(f'finished saving')
    time.sleep(100)
    os.remove(f'{filename}.zip')
