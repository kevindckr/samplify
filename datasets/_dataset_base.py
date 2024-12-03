import os
import sys
import numpy as np

from enum import Enum, auto
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


class BaseImageDataset(ABC):
    """
    A base class that ingests a data directory and loads an image and its pixel labels.
    """
    def __init__(self, data_dir: str, name: str, 
                 classes: Dict[int, str], colors: Dict[int, str], wavelength_range: Tuple[int, int]) -> None:
        self.name: str = name
        self.data_dir: str = data_dir
        self.classes: Dict[int, str] = classes
        self.colors: Dict[int, str] = colors
        self.wavelength_range: Tuple[int, int] = wavelength_range
        self.image: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None

        # Ensure class indices are well formatted.
        # NOTE: unlabelled must be present at index 0, but it does not have to be used! (i.e. everything can be labelled)
        # If you want to ignore other classes, zero those labels out before loading into this class!
        class_indices = sorted(self.classes.keys())
        if class_indices != list(range(len(class_indices))) or (0 in self.classes and self.classes[0] != 'unlabelled'):
            raise ValueError(f'Class indices must start at 0, be contiguous, and \'unlabelled\' must be at index 0.')

        # TODO: Add a check on the types of image and labels; float32 and int32

    @abstractmethod
    def load(self) -> None:
        pass

    # "Bandmax" normalization for hyperspectral data
    # https://arxiv.org/ftp/arxiv/papers/1710/1710.02939.pdf
    def bandmax_normalization(self):
        print(f'> Bandmax normalization...')
        band_maximums = np.max(self.image, axis=(0, 1))
        self.image = self.image / band_maximums

    def minmax_normalization(self):
        print(f'> minmax normalization...')
        self.image = (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image))

    def rgb_view_bands(self) -> Tuple[int, int, int]:
        # Define the typical wavelengths for R, G, B
        wavelengths = {
            'blue': 450,
            'green': 550,
            'red': 650
        }
        
        # Calculate the width of each band
        band_width = (self.wavelength_range[1] - self.wavelength_range[0]) / self.image.shape[-1]
        
        # Helper function to calculate index
        def get_band_index(wavelength):
            index = int((wavelength - self.wavelength_range[0]) / band_width)
            if index < 0:
                return 0
            elif index >= self.image.shape[-1]:
                return self.image.shape[-1] - 1
            return index

        # Calculate the indices for R, G, B and ensure they are within the valid range
        blue_index = get_band_index(wavelengths['blue'])
        green_index = get_band_index(wavelengths['green'])
        red_index = get_band_index(wavelengths['red'])

        return red_index, green_index, blue_index
