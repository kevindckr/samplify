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


class DatasetBotswana(BaseImageDataset):

    classes = {
        0: 'unlabelled',
        1: 'water',
        2: 'hippo_grass',
        3: 'floodplain_grasses_1',
        4: 'floodplain_grasses_2',
        5: 'reeds',
        6: 'riparian',
        7: 'fire_scar',
        8: 'island_interior',
        9: 'acacia_woodlands',
        10: 'acacia_shrublands',
        11: 'acacia_grasslands',
        12: 'short_mopane',
        13: 'mixed_mopane',
        14: 'exposed_soils'
    }

    colors = {
        0: 'black',                 # unlabelled
        1: 'dodgerblue',            # water
        2: 'forestgreen',           # hippo_grass
        3: 'darkgreen',             # floodplain_grasses_1
        4: 'limegreen',             # floodplain_grasses_2
        5: 'mediumseagreen',        # reeds
        6: 'olive',                 # riparian
        7: 'darkred',               # fire_scar
        8: 'tan',                   # island_interior
        9: 'saddlebrown',           # acacia_woodlands
        10: 'peru',                 # acacia_shrublands
        11: 'goldenrod',            # acacia_grasslands
        12: 'darkorange',           # short_mopane
        13: 'sienna',                # mixed_mopane
        14: 'sandybrown'            # exposed_soils
    }

    wavelength = [400, 2500]  # EO-1 Hyperion

    def __init__(self, data_dir: str) -> None:
        data_dir = os.path.join(data_dir, 'botswana')
        super().__init__(data_dir, 'Botswana', DatasetBotswana.classes, DatasetBotswana.colors, DatasetBotswana.wavelength)

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
        mat_path = os.path.join(self.data_dir, 'Botswana.mat')
        mat_data = scipy.io.loadmat(mat_path)
        total_hyper = mat_data['Botswana']
        print(f'    Hyper Shape: {total_hyper.shape}')
        return total_hyper
    
    def _load_label(self) -> np.ndarray:
        print('  Loading labels MAT...')
        mat_path = os.path.join(self.data_dir, 'Botswana_gt.mat')
        mat_data = scipy.io.loadmat(mat_path)
        total_labels = mat_data['Botswana_gt']
        print(f'    Labels Shape: {total_labels.shape}')
        return total_labels
