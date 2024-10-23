import torch
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_single_tensor(t):
    """
    Plots a single tensor and saves it to the path with the given name.
    :param tensor: The tensor to plot.
    :param path: The path to save the image.
    :param name: The name of the image.
    :return: None
    """

    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    
    if len(t.shape) == 3:
        t = t[0]
    
    # plot the numpy array
    plt.imshow(t, cmap='gray')
    # save the image
    plt.imsave('example_tensor.png', t)

