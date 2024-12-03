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


class DatasetPaviaUniversity(BaseImageDataset):

    classes = {
        0: 'unlabelled',
        1: 'asphalt',
        2: 'meadows',
        3: 'gravel',
        4: 'trees',
        5: 'painted_metal_sheets',
        6: 'bare_soil',
        7: 'bitumen',
        8: 'self_blocking_bricks',
        9: 'shadows'
    }

    colors = {
        0: 'black',                 # unlabelled
        1: 'dimgray',               # asphalt
        2: 'limegreen',             # meadows
        3: 'rosybrown',             # gravel
        4: 'forestgreen',           # trees
        5: 'skyblue',               # painted_metal_sheets
        6: 'sandybrown',            # bare_soil
        7: 'darkorange',            # bitumen
        8: 'sienna',                # self_blocking_bricks
        9: 'darkslategray'          # shadows
    }

    wavelength = [430, 860]  # ROSIS

    def __init__(self, data_dir: str) -> None:
        data_dir = os.path.join(data_dir, 'pavia_university')
        super().__init__(data_dir, 'PaviaUniversity', DatasetPaviaUniversity.classes, DatasetPaviaUniversity.colors, DatasetPaviaUniversity.wavelength)

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
        mat_path = os.path.join(self.data_dir, 'PaviaU.mat')
        mat_data = scipy.io.loadmat(mat_path)
        total_hyper = mat_data['paviaU']
        print(f'    Hyper Shape: {total_hyper.shape}')
        return total_hyper
    
    def _load_label(self) -> np.ndarray:
        print('  Loading labels MAT...')
        mat_path = os.path.join(self.data_dir, 'PaviaU_gt.mat')
        mat_data = scipy.io.loadmat(mat_path)
        total_labels = mat_data['paviaU_gt']
        print(f'    Labels Shape: {total_labels.shape}')
        return total_labels
