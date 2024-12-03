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


class DatasetSalinas(BaseImageDataset):

    classes = {
        0: 'unlabelled',
        1: 'brocoli_green_weeds_1',
        2: 'brocoli_green_weeds_2',
        3: 'fallow',
        4: 'fallow_rough_plow',
        5: 'fallow_smooth',
        6: 'stubble',
        7: 'celery',
        8: 'grapes_untrained',
        9: 'soil_vinyard_develop',
        10: 'corn_senesced_green_weeds',
        11: 'lettuce_romaine_4wk',
        12: 'lettuce_romaine_5wk',
        13: 'lettuce_romaine_6wk',
        14: 'lettuce_romaine_7wk',
        15: 'vinyard_untrained',
        16: 'vinyard_vertical_trellis'
    }

    colors = {
        0: 'black',                 # unlabelled
        1: 'limegreen',             # brocoli_green_weeds_1
        2: 'forestgreen',           # brocoli_green_weeds_2
        3: 'khaki',                 # fallow
        4: 'tan',                   # fallow_rough_plow
        5: 'wheat',                 # fallow_smooth
        6: 'lightgray',             # stubble
        7: 'darkgreen',             # celery
        8: 'purple',                # grapes_untrained
        9: 'sandybrown',            # soil_vinyard_develop
        10: 'goldenrod',            # corn_senesced_green_weeds
        11: 'olivedrab',            # lettuce_romaine_4wk
        12: 'yellowgreen',          # lettuce_romaine_5wk
        13: 'mediumseagreen',       # lettuce_romaine_6wk
        14: 'seagreen',             # lettuce_romaine_7wk
        15: 'olive',                # vinyard_untrained
        16: 'darkolivegreen'        # vinyard_vertical_trellis
    }

    wavelength = [400, 2500]  # AVIRIS

    def __init__(self, data_dir: str) -> None:
        data_dir = os.path.join(data_dir, 'salinas')
        super().__init__(data_dir, 'Salinas', DatasetSalinas.classes, DatasetSalinas.colors, DatasetSalinas.wavelength)

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
        mat_path = os.path.join(self.data_dir, 'Salinas.mat')
        mat_data = scipy.io.loadmat(mat_path)
        total_hyper = mat_data['salinas']
        print(f'    Hyper Shape: {total_hyper.shape}')
        return total_hyper
    
    def _load_label(self) -> np.ndarray:
        print('  Loading labels MAT...')
        mat_path = os.path.join(self.data_dir, 'Salinas_gt.mat')
        mat_data = scipy.io.loadmat(mat_path)
        total_labels = mat_data['salinas_gt']
        print(f'    Labels Shape: {total_labels.shape}')
        return total_labels
