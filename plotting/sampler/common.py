import os
import sys
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from datasets._dataset_base import BaseImageDataset
from samplers._sampler_base import BaseImageSampler
from samplers._overlap import cpl_validity_numba

FIGURE_DPI = 300
FIGSIZE_RECT = (24, 12)
FIGSIZE_SQUARE = (24, 24)


def rotate_to_horizontal(image_data: np.ndarray) -> np.ndarray:
    if image_data.shape[0] > image_data.shape[1]:
        return np.rot90(image_data)
    return image_data


def dataset_colormap(dataset: BaseImageDataset):
    # Create a colormap for the labels
    colors = dataset.colors
    label_colormap = ListedColormap(list(colors.values()))
    label_min, label_max, label_cnt = min(colors.keys()), max(colors.keys()), len(colors)
    return label_colormap, label_min, label_max, label_cnt


def pixel_status_colormap():
    # Create a color map for the pixel status
    #     status_colors = {
    #     0: 'black',         # Unlabelled
    #     1: 'yellow',        # Overlapping Patch
    #     2: 'lawngreen',     # Train Patch Valid
    #     3: 'red',           # Test  Patch Valid
    #     4: 'darkgreen',     # Train CPL Valid
    #     5: 'lightgreen',    # Train CPL Invalid
    #     6: 'darkred',       # Test  CPL Valid
    #     7: 'lightcoral',    # Test  CPL Invalid
    #     8: 'white'          # -
    # }
    status_colors = {
        0: 'black',         # Unlabelled
        1: 'darkred',       # Overlapping Patch ( 139, 0, 0 )
        2: 'ForestGreen',   # Train Patch Valid ( 34, 139, 34 )
        3: 'LightGreen',    # Test  Patch Valid ( 144, 238, 144 )
        4: 'DarkGreen',     # Train CPL Valid   ( 0, 100, 0 )
        5: 'Red',           # Train CPL Invalid ( 255, 0, 0 )
        6: 'LawnGreen',     # Test  CPL Valid   ( 124, 252, 0 )
        7: 'LightCoral',    # Test  CPL Invalid ( 240, 128, 128 )
        8: 'white'          # -
    }

    status_colormap = ListedColormap(list(status_colors.values()))
    status_min, status_max, status_cnt = min(status_colors.keys()), max(status_colors.keys()), len(status_colors)

    return status_colormap, status_min, status_max, status_cnt


def create_cpl_plot_data(dataset: BaseImageDataset, sampler: BaseImageSampler):
    # Pull out the information we need
    image = dataset.image
    labels = dataset.labels
    train_cpls = sampler.training_cpls
    test_cpls = sampler.testing_cpls
    patch_bboxes = sampler.patch_bboxes

    # Create the training and testing labels image
    train_labels = np.zeros_like(labels)
    for class_idx, class_cpls in train_cpls.items():
        for (y, x) in class_cpls:
            train_labels[y, x] = class_idx

    test_labels = np.zeros_like(labels)
    for class_idx, class_cpls in test_cpls.items():
        for (y, x) in class_cpls:
            test_labels[y, x] = class_idx

    # Calculate all of information about pixel validity
    flat_train_cpls = np.concatenate(list(train_cpls.values()), axis=0)
    flat_test_cpls = np.concatenate(list(test_cpls.values()), axis=0)
    invalid_pixels, valid_train_pixels, valid_train_cpls, invalid_train_cpls, valid_test_pixels, valid_test_cpls, invalid_test_cpls = cpl_validity_numba(image.shape, flat_train_cpls, flat_test_cpls, patch_bboxes)

    # Create the footprints image
    footprint_image = np.zeros_like(labels)
    footprint_image[invalid_pixels[:, 0], invalid_pixels[:, 1]] = 1
    footprint_image[valid_train_pixels[:, 0], valid_train_pixels[:, 1]] = 2
    footprint_image[valid_test_pixels[:, 0], valid_test_pixels[:, 1]] = 3
    # footprint_image[unused_valid_cpls[:, 0], unused_valid_cpls[:, 1]] = 8

    # Create the footprints+cpls image
    cpls_image = np.zeros_like(labels)
    cpls_image[invalid_pixels[:, 0], invalid_pixels[:, 1]] = 1
    cpls_image[valid_train_pixels[:, 0], valid_train_pixels[:, 1]] = 2
    cpls_image[valid_test_pixels[:, 0], valid_test_pixels[:, 1]] = 3
    cpls_image[valid_train_cpls[:, 0], valid_train_cpls[:, 1]] = 4
    cpls_image[invalid_train_cpls[:, 0], invalid_train_cpls[:, 1]] = 5
    cpls_image[valid_test_cpls[:, 0], valid_test_cpls[:, 1]] = 6
    cpls_image[invalid_test_cpls[:, 0], invalid_test_cpls[:, 1]] = 7
    # cpls_image[unused_valid_cpls[:, 0], unused_valid_cpls[:, 1]] = 8

    # Potentially rotate all images to ensure longest side is horizontal
    image = rotate_to_horizontal(image)
    labels = rotate_to_horizontal(labels)
    test_labels = rotate_to_horizontal(test_labels)
    train_labels = rotate_to_horizontal(train_labels)
    cpls_image = rotate_to_horizontal(cpls_image)
    footprint_image = rotate_to_horizontal(footprint_image)

    # Select image bands and normalize
    image = image[:, :, dataset.rgb_view_bands()]
    image = (image - np.min(image)) / np.ptp(image)

    return image, labels, test_labels, train_labels, footprint_image, cpls_image


def plot_cpls(dataset: BaseImageDataset, sampler: BaseImageSampler, save_or_show: str = None):
    # Get the colormaps and data for the plot
    label_colormap, label_min, label_max, label_cnt = dataset_colormap(dataset)
    status_colormap, status_min, status_max, status_cnt = pixel_status_colormap()
    image, labels, test_labels, train_labels, footprint_image, cpls_image = create_cpl_plot_data(dataset, sampler)

    # Plot the image and labels
    fig, axs = plt.subplots(3, 2, figsize=FIGSIZE_RECT)

    # Plot the image
    ax = axs[0, 0]
    ax.imshow(image)
    ax.axis('off')
    ax.set_title('Image')

    # Plot the raw labels
    ax = axs[1, 0]
    ax.imshow(labels, cmap=label_colormap, interpolation='none', vmin=label_min, vmax=label_max)
    ax.axis('off')
    ax.set_title('Labels')
    
    # Plot the train labels
    ax = axs[0, 1]
    ax.imshow(train_labels, cmap=label_colormap, interpolation='none', vmin=label_min, vmax=label_max)
    ax.axis('off')
    ax.set_title('Training CPLs')

    # Plot the test labels
    ax = axs[1, 1]
    ax.imshow(test_labels, cmap=label_colormap, interpolation='none', vmin=label_min, vmax=label_max)
    ax.axis('off')
    ax.set_title('Testing CPLs')

    # Plot the footprint of just patch pixels, not cpls
    ax = axs[2, 0]
    ax.imshow(footprint_image, cmap=status_colormap, interpolation='none', vmin=status_min, vmax=status_max)
    #ax.axis('off')
    ax.set_title('Train, Test, Overlap Footprint')
    ax.set_xticks(np.arange(-0.5, image.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, image.shape[0], 1), minor=True)
    ax.grid(True, which='minor', color='white', linestyle='-', linewidth=0.15)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Plot the footprint of patch and cpls 
    ax = axs[2, 1]
    ax.imshow(cpls_image, cmap=status_colormap, interpolation='none', vmin=status_min, vmax=status_max)
    #ax.axis('off')
    ax.set_title('Train, Test, Overlap Footprint and CPLs')
    ax.set_xticks(np.arange(-0.5, image.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, image.shape[0], 1), minor=True)
    ax.grid(True, which='minor', color='white', linestyle='-', linewidth=0.15)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Tight layout and save or show
    plt.tight_layout()
    if save_or_show:
        plt.savefig(save_or_show, dpi=FIGURE_DPI)
        plt.close()
    else:
        plt.show()
