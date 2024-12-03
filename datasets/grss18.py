import os
import sys
import numpy as np
from PIL import Image

from enum import Enum, auto
from typing import Optional, Tuple

from spectral import *

from datasets._dataset_base import BaseImageDataset


class DatasetGRSS18(BaseImageDataset):

    classes = {
        0: 'unlabelled',
        1: 'healthy_grass',
        2: 'stressed_grass',
        3: 'artificial_turf',
        4: 'evergreen_trees',
        5: 'deciduous_trees',
        6: 'bare_earth',
        7: 'water',
        8: 'residential_buildings',
        9: 'non_residential_buildings',
        10: 'roads',
        11: 'sidewalks',
        12: 'crosswalks',
        13: 'major_thoroughfares',
        14: 'highways',
        15: 'railways',
        16: 'paved_parking_lots',
        17: 'unpaved_parking_lots',
        18: 'cars',
        19: 'trains',
        20: 'stadium_seats'
    }

    colors = {
        0: 'black',             # unlabelled
        1: 'green',             # healthy_grass
        2: 'darkgreen',         # stressed_grass
        3: 'lime',              # artificial_turf
        4: 'cadetblue',         # evergreen_trees
        5: 'steelblue',         # deciduous_trees
        6: 'olive',             # bare_earth
        7: 'lightblue',         # water
        8: 'plum',              # residential_buildings
        9: 'rebeccapurple',     # non-residential_buildings
        10: 'darkgrey',         # roads
        11: 'dimgrey',          # sidewalks
        12: 'lightgrey',        # crosswalks
        13: 'wheat',            # major_thoroughfares
        14: 'lightcoral',       # highways
        15: 'deeppink',         # railways
        16: 'midnightblue',     # paved_parking_lots
        17: 'dodgerblue',       # unpaved_parking_lots
        18: 'red',              # cars
        19: 'chocolate',        # trains
        20: 'yellow',           # stadium_seats
    }

    wavelength = [380, 1050]  # Sensor?

    def __init__(self, data_dir: str) -> None:
        data_dir = os.path.join(data_dir, 'grss_18')
        super().__init__(data_dir, 'GRSS18', DatasetGRSS18.classes, DatasetGRSS18.colors, DatasetGRSS18.wavelength)

    def load(self) -> None:
        print(f'> Loading dataset: {self.name}')
        try:
            _, image = self._load_hyper()
            _, labels = self._load_label()
        except Exception as e:
            raise RuntimeError('Could not load dataset:\nReason: ' + str(e))

        self.image = image
        self.labels = labels
    
    def _load_hyper(self) -> Tuple[np.ndarray, np.ndarray]:
        print(f'  Loading hyperspectral ENVI...')
        hdr_path = os.path.join(self.data_dir, '20170218_UH_CASI_S4_NAD83.hdr')  # 'FullHSIDataset'
        pix_path = os.path.join(self.data_dir, '20170218_UH_CASI_S4_NAD83.pix')
        raw_hyper_envi = io.envi.open(hdr_path, pix_path)
        total_hyper = np.copy(raw_hyper_envi[:, :, 0:48])
        label_hyper = np.copy(raw_hyper_envi[601:1202, 596:2980, 0:48])  # Slice only the labelled pixels.

        print(f'    Hyper Shape (Total): {total_hyper.shape}')
        print(f'    Hyper Shape (Train): {label_hyper.shape}')
        return total_hyper, label_hyper
    
    def _load_label(self) -> Tuple[np.ndarray, np.ndarray]:
        print('  Loading labels ENVI...')
        raw_path = os.path.join(self.data_dir, '2018_IEEE_GRSS_DFC_GT_TR')  # TrainingGT
        hdr_path = os.path.join(self.data_dir, '2018_IEEE_GRSS_DFC_GT_TR.hdr')
        raw_label_envi = io.envi.open(hdr_path, raw_path)
        total_labels = np.copy(np.squeeze(raw_label_envi[:, :, 0]))
        down_labels = self.max_pool_2d(total_labels)
        # Force to integer type.
        total_labels = total_labels.astype(np.int32)
        down_labels = down_labels.astype(np.int32)

        print(f'    Labels Shape (Raw ): {total_labels.shape}')
        print(f'    Labels Shape (Down): {down_labels.shape}')
        return total_labels, down_labels

    @staticmethod
    def max_pool_2d(labels, pool_size=(2, 2)):
        # Check if the labels size is divisible by pool_size, pad otherwise
        pad_height = (pool_size[0] - labels.shape[0] % pool_size[0]) % pool_size[0]
        pad_width = (pool_size[1] - labels.shape[1] % pool_size[1]) % pool_size[1]
        labels_padded = np.pad(labels, ((0, pad_height), (0, pad_width)), mode='constant')

        # Reshape the array to group pool_size blocks together
        pooled = labels_padded.reshape(labels_padded.shape[0] // pool_size[0], pool_size[0], labels_padded.shape[1] // pool_size[1], pool_size[1])

        pooled = pooled.max(axis=(1, 3))
        return pooled
    