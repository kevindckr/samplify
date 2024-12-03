import os
import sys
import numpy as np
from PIL import Image

from enum import Enum, auto
from typing import Optional, Tuple

from spectral import *
from PIL import Image

from datasets._dataset_base import BaseImageDataset


class DatasetGRSS13(BaseImageDataset):

    classes = {
        0: 'unlabelled',
        1: 'healthy_grass',
        2: 'stressed_grass',
        3: 'artificial_turf',
        4: 'trees',
        #5: 'deciduous_trees', 4: 'evergreen_trees',
        5: 'bare_earth',
        6: 'water',
        7: 'residential',
        8: 'commercial',
        9: 'road',
        10: 'highway',
        11: 'railway',
        12: 'parking_lot_1',
        13: 'parking_lot_2',
        14: 'tennis_court',
        15: 'running_track'
    }

    colors = {
        0: 'black',             # unlabelled
        1: 'green',             # healthy_grass
        2: 'darkgreen',         # stressed_grass
        3: 'lime',              # artificial_turf
        4: 'cadetblue',         # evergreen_trees
        #5: 'steelblue',        # deciduous_trees
        5: 'olive',             # bare_earth
        6: 'lightblue',         # water
        7: 'plum',              # residential
        8: 'rebeccapurple',     # commercial
        9: 'darkgrey',         # road
        10: 'dimgrey',          # highway
        11: 'lightgrey',        # railway
        12: 'wheat',            # parking_lot_1
        13: 'lightcoral',       # parking_lot_2
        14: 'deeppink',         # tennis_court
        15: 'midnightblue',     # running_track
    }

    wavelength = [380, 1050]  # Sensor?

    def __init__(self, data_dir: str) -> None:
        data_dir = os.path.join(data_dir, 'grss_13')
        super().__init__(data_dir, 'GRSS13', DatasetGRSS13.classes, DatasetGRSS13.colors, DatasetGRSS13.wavelength)

        # The IEEE GRSS DFC suggested training and testing labels.
        self.labels_train: Optional[np.ndarray] = None
        self.labels_test: Optional[np.ndarray] = None

    def load(self) -> None:
        print(f'> Loading dataset: {self.name}')
        try:
            image = self._load_hyper()
            labels, labels_train, labels_test = self._load_label()
        except Exception as e:
            raise RuntimeError('Could not load dataset:\nReason: ' + str(e))

        self.image = image
        self.labels = labels
        self.labels_train = labels_train
        self.labels_test = labels_test
    
    def _load_hyper(self) -> np.ndarray:
        print(f'  Loading hyperspectral ENVI...')
        hdr_path = os.path.join(self.data_dir, '2013_IEEE_GRSS_DF_Contest_CASI.hdr')
        tif_path = os.path.join(self.data_dir, '2013_IEEE_GRSS_DF_Contest_CASI.tif')
        raw_hyper_envi = io.envi.open(hdr_path, tif_path)
        total_hyper = np.copy(raw_hyper_envi[:, :, 0:48])
        print(f'    Hyper Shape: {total_hyper.shape}')
        return total_hyper
    
    def _load_label(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        print('  Loading labels TIF...')
        train_tif_path = os.path.join(self.data_dir, 'train_roi.tif')  # 'labels_tiff_alternate_source'
        val_tif_path = os.path.join(self.data_dir, 'val_roi.tif')
        labels_train = np.array(Image.open(train_tif_path))
        labels_val = np.array(Image.open(val_tif_path))
        # Add the pre-sampled train and validation labels to just get all labels, we are doing the sampling!
        total_labels = labels_train + labels_val

        # Check to ensure whomever made these tifs did it correctly.
        condition1 = np.logical_and(labels_train > 0, labels_val == 0).any()
        condition2 = np.logical_and(labels_val > 0, labels_train == 0).any()
        if not condition1 or not condition2:
            raise AssertionError(f'Train and Validation .tif labels are not mutually exclusive!')

        print(f'    Labels Shape: {total_labels.shape}')
        return total_labels, labels_train, labels_val
