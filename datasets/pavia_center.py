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


class DatasetPaviaCenter(BaseImageDataset):

    classes = {
        0: 'unlabelled',
        1: 'water',
        2: 'trees',
        3: 'asphalt',
        4: 'self_blocking_bricks',
        5: 'bitumen',
        6: 'tiles',
        7: 'shadows',
        8: 'meadows',
        9: 'bare_soil'
    }

    colors = {
        0: 'black',              # unlabelled
        1: 'deepskyblue',        # water
        2: 'forestgreen',        # trees
        3: 'dimgray',            # asphalt
        4: 'sienna',             # self_blocking_bricks
        5: 'darkorange',         # bitumen
        6: 'lightgray',          # tiles
        7: 'darkslategray',      # shadows
        8: 'limegreen',          # meadows
        9: 'sandybrown'          # bare_soil
    }

    wavelength = [430, 860]  # ROSIS

    def __init__(self, data_dir: str) -> None:
        data_dir = os.path.join(data_dir, 'pavia_center')
        super().__init__(data_dir, 'PaviaCenter', DatasetPaviaCenter.classes, DatasetPaviaCenter.colors, DatasetPaviaCenter.wavelength)

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
        mat_path = os.path.join(self.data_dir, 'Pavia.mat')
        mat_data = scipy.io.loadmat(mat_path)
        total_hyper = mat_data['pavia']
        print(f'    Hyper Shape: {total_hyper.shape}')
        return total_hyper
    
    def _load_label(self) -> np.ndarray:
        print('  Loading labels MAT...')
        mat_path = os.path.join(self.data_dir, 'Pavia_gt.mat')
        mat_data = scipy.io.loadmat(mat_path)
        total_labels = mat_data['pavia_gt']
        print(f'    Labels Shape: {total_labels.shape}')
        return total_labels
