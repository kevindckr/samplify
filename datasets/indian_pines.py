import os
import sys
import scipy
import numpy as np
from PIL import Image

from enum import Enum, auto
from typing import Optional, Tuple

from spectral import *
from PIL import Image

from datasets._dataset_base import BaseImageDataset


class DatasetIndianPines(BaseImageDataset):

    classes = {
        0: 'unlabelled',
        1: 'alfalfa',
        2: 'corn_notill',
        3: 'corn_mintill',
        4: 'corn',
        5: 'grass_pasture',
        6: 'grass_trees',
        7: 'grass_pasture_mowed',
        8: 'hay_windrowed',
        9: 'oats',
        10: 'soybean_notill',
        11: 'soybean_mintill',
        12: 'soybean_clean',
        13: 'wheat',
        14: 'woods',
        15: 'buildings_grass_trees_drives',
        16: 'stone_steel_towers'
    }

    colors = {
        0: 'black',             # unlabelled
        1: 'darkviolet',        # alfalfa
        2: 'gold',              # corn_notill
        3: 'sienna',            # corn_mintill
        4: 'turquoise',         # corn
        5: 'lightsalmon',       # grass_pasture
        6: 'mediumseagreen',    # grass_trees
        7: 'navy',              # grass_pasture_mowed
        8: 'sandybrown',        # hay_windrowed
        9: 'darkslategray',     # oats
        10: 'maroon',           # soybean_notill
        11: 'olive',            # soybean_mintill
        12: 'orchid',           # soybean_clean
        13: 'lightseagreen',    # wheat
        14: 'indigo',           # woods
        15: 'teal',             # buildings_grass_trees_drives
        16: 'peru',             # stone_steel_towers
    }

    wavelength = [400, 2500]  # AVIRIS

    def __init__(self, data_dir: str) -> None:
        data_dir = os.path.join(data_dir, 'indian_pines')
        super().__init__(data_dir, 'IndianPines', DatasetIndianPines.classes, DatasetIndianPines.colors, DatasetIndianPines.wavelength)

    def load(self) -> None:
        print(f'> Loading dataset: {self.name}')
        try:
            image = self._load_hyper()
            labels = self._load_label()
        except Exception as e:
            raise RuntimeError('Could not load dataset:\nReason: ' + str(e))

        self.image = image
        self.labels = labels
    
    def _load_hyper(self) -> np.ndarray:
        print(f'  Loading hyperspectral MAT...')
        mat_path = os.path.join(self.data_dir, 'Indian_pines.mat')
        mat_data = scipy.io.loadmat(mat_path)
        total_hyper = mat_data['indian_pines']
        print(f'    Hyper Shape: {total_hyper.shape}')
        return total_hyper
    
    def _load_label(self) -> np.ndarray:
        print('  Loading labels MAT...')
        mat_path = os.path.join(self.data_dir, 'Indian_pines_gt.mat')
        mat_data = scipy.io.loadmat(mat_path)
        total_labels = mat_data['indian_pines_gt']
        print(f'    Labels Shape: {total_labels.shape}')
        return total_labels
