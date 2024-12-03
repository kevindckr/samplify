import os
import sys
import random
import numpy as np
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

FIGSIZE_RECT = (24, 12)
FIGSIZE_SQUARE = (24, 24)


def rotate_to_horizontal(image_data: np.ndarray) -> np.ndarray:
    if image_data.shape[0] > image_data.shape[1]:
        return np.rot90(image_data)
    return image_data

def plot_image(image: np.ndarray, bands: Tuple[int, int, int] = None, save_or_show: str = None):
    # Potentially rotate the image to ensure longest side is horizontal
    image = rotate_to_horizontal(image)

    # Select 3 bands as RGB if not specified
    if bands is None:
        bands = random.sample(range(image.shape[2]), 3)
    image = image[:, :, bands]
    
    # Normalize image
    image = (image - np.min(image)) / np.ptp(image)

    # Plot the image
    fig = plt.figure(figsize=FIGSIZE_RECT)
    plt.imshow(image)
    #plt.axis('off')
    
    if save_or_show:
        plt.savefig(save_or_show)
    else:
        plt.show()

def plot_labels(labels: np.ndarray, colors: Dict[int, str], classes: Dict[int, str], save_or_show: str = None):
    # Potentially rotate the image to ensure longest side is horizontal
    labels = rotate_to_horizontal(labels)

    # Create a colormap for the labels
    cm = ListedColormap(list(colors.values()))
    l_min = min(colors.keys())  # Use keys for minimum and maximum labels
    l_max = max(colors.keys())
    l_cnt = len(colors)

    # Plot the image
    fig = plt.figure(figsize=FIGSIZE_RECT)
    cax = plt.imshow(labels, cmap=cm, interpolation=None, vmin=l_min, vmax=l_max)
    cbar = fig.colorbar(cax, ticks=np.linspace(l_min, l_max, 2 * l_cnt + 1)[1::2], shrink=0.45)
    cbar.ax.set_yticklabels(list(classes.values()))  # Use list() to ensure the order is preserved

    if save_or_show:
        plt.savefig(save_or_show)
    else:
        plt.show()

def plot_image_labels(image: np.ndarray, labels: np.ndarray, colors: Dict[int, str], bands: Tuple[int, int, int] = None, save_or_show: str = None):
    # Potentially rotate the image and labels to ensure longest side is horizontal
    image = rotate_to_horizontal(image)
    labels = rotate_to_horizontal(labels)

    # Select 3 bands as RGB if not specified
    if bands is None:
        bands = random.sample(range(image.shape[2]), 3)
    image = image[:, :, bands]
    
    # Normalize image
    image = (image - np.min(image)) / np.ptp(image)

    # Create a colormap for the labels
    cm = ListedColormap(colors.values())
    l_min = min(colors)
    l_max = max(colors)
    l_cnt = len(colors)

    # Plot the image and labels
    fig, axs = plt.subplots(2, 1, figsize=FIGSIZE_RECT)
    
    # Plot the image
    axs[0].imshow(image)
    axs[0].axis('off')
    #axs[0].set_title('Image')

    # Plot the labels
    cax = axs[1].imshow(labels, cmap=cm, interpolation=None, vmin=l_min, vmax=l_max)
    axs[1].axis('off')
    #axs[1].set_title('Labels')

    plt.tight_layout()

    if save_or_show:
        plt.savefig(save_or_show)
    else:
        plt.show()
