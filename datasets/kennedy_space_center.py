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


class DatasetKennedySpaceCenter(BaseImageDataset):

    classes = {
        0: 'unlabelled',
        1: 'scrub',
        2: 'willow_swamp',
        3: 'cp_hammock',
        4: 'slash_pine',
        5: 'oak_broadleaf',
        6: 'hardwood',
        7: 'swamp',
        8: 'graminoid_marsh',
        9: 'spartina_marsh',
        10: 'cattail_marsh',
        11: 'salt_marsh',
        12: 'mud_flats',
        13: 'water',
    }

    colors = {
        0: 'black',                 # unlabelled
        1: 'forestgreen',           # scrub
        2: 'olivedrab',             # willow_swamp
        3: 'saddlebrown',           # cp_hammock
        4: 'sienna',                # slash_pine
        5: 'darkorange',            # oak_broadleaf
        6: 'peru',                  # hardwood
        7: 'darkslategray',         # swamp
        8: 'lightgreen',            # graminoid_marsh
        9: 'limegreen',             # spartina_marsh
        10: 'darkseagreen',         # cattail_marsh
        11: 'lightcyan',            # salt_marsh
        12: 'sandybrown',           # mud_flats
        13: 'dodgerblue'            # water
    }

    wavelength = [400, 2500]  # AVIRIS

    def __init__(self, data_dir: str) -> None:
        data_dir = os.path.join(data_dir, 'ksc')
        super().__init__(data_dir, 'KennedySpaceCenter', DatasetKennedySpaceCenter.classes, DatasetKennedySpaceCenter.colors, DatasetKennedySpaceCenter.wavelength)

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
        mat_path = os.path.join(self.data_dir, 'KSC_corrected.mat')
        mat_data = scipy.io.loadmat(mat_path)
        total_hyper = mat_data['KSC']
        print(f'    Hyper Shape: {total_hyper.shape}')
        return total_hyper
    
    def _load_label(self) -> np.ndarray:
        print('  Loading labels MAT...')
        mat_path = os.path.join(self.data_dir, 'KSC_gt.mat')
        mat_data = scipy.io.loadmat(mat_path)
        total_labels = mat_data['KSC_gt']
        print(f'    Labels Shape: {total_labels.shape}')
        return total_labels
