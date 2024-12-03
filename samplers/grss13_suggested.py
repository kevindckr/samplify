import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

from datasets.grss13 import DatasetGRSS13

from datasets._dataset_base import BaseImageDataset
from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerGRSS13Suggested(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: True
    
    Article Bibtex Reference:
        @article{Debes2014,
            abstract = {The 2013 Data Fusion Contest organized by the Data Fusion Technical Committee (DFTC) of the IEEE Geoscience and Remote 
            Sensing Society aimed at investigating the synergistic use of hyperspectral and Light Detection And Ranging (LiDAR) data. The data 
            sets distributed to the participants during the Contest, a hyperspectral imagery and the corresponding LiDAR-derived digital surface 
            model (DSM), were acquired by the NSF-funded Center for Airborne Laser Mapping over the University of Houston campus and its 
            neighboring area in the summer of 2012. This paper highlights the two awarded research contributions, which investigated different 
            approaches for the fusion of hyperspectral and LiDAR data, including a combined unsupervised and supervised classification scheme, 
            and a graph-based method for the fusion of spectral, spatial, and elevation information. © 2014 IEEE.},
            author = {Christian Debes and Andreas Merentitis and Roel Heremans and Jurgen Hahn and Nikolaos Frangiadakis 
            and Tim Van Kasteren and Wenzhi Liao and Rik Bellens and Aleksandra Pizurica and Sidharta Gautama and Wilfried Philips 
            and Saurabh Prasad and Qian Du and Fabio Pacifici},
            doi = {10.1109/JSTARS.2014.2305441},
            issn = {21511535},
            issue = {6},
            journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
            pages = {2405-2418},
            publisher = {Institute of Electrical and Electronics Engineers},
            title = {Hyperspectral and LiDAR data fusion: Outcome of the 2013 GRSS data fusion contest},
            volume = {7},
            year = {2014},
        }

    Article Sampler Description Excerpt:
    N/A

    Observations/Notes:
    - Many articles which utilize the GRSS13 data note that they use the suggested/provided training and testing CPLs provided by the contest
    committee. This sampler then provides exactly that set of the training and testing CPLs. The only other action this method performs is to
    ensure that the provided CPLs do not allow patches to overrun the edge of the image bounds because it allows for variable patch sizing.
    """
    def __init__(self,
                 image: np.ndarray, labels: np.ndarray, classes: Dict[int, str],
                 patch_size: Tuple[int, int], train_ratio: float, 
                 name: str, random_seed: Optional[int] = None, **kwargs) -> None:
        super().__init__(image, labels, classes, patch_size, train_ratio, name, random_seed, **kwargs)
        
        # The labels split between training and testing as provided by the IEEE GRSS13 DFC committee
        self.labels_train: np.ndarray = kwargs.get('labels_train', None)
        self.labels_test: np.ndarray = kwargs.get('labels_test', None)

        # Ensure user is only sampling on GRSS13 data and not another dataset
        if self.labels_train is None or self.labels_test is None:
            raise RuntimeError(f'! Must provide training and testing labels. Train: {self.labels_train is not None} Test: {self.labels_test is not None} !')
        self.dataset: BaseImageDataset = kwargs.get('dataset', None)
        if self.dataset is not None and not isinstance(self.dataset, DatasetGRSS13):
            raise RuntimeError(f'! Can only instantiate {self.__class__.__name__} from {DatasetGRSS13.__class__.__name__} !')
        if self.dataset is None and self.image.shape[:2] != self.labels_train.shape:
            raise RuntimeError(f'! Can only instantiate {self.__class__.__name__} from GRSS13 image !')

    @numpy_deterministic
    def sample(self) -> None:
        print(f'> Executing: {self.name} ({self.__class__.__name__})')

        # Get the center pixel locations as provided for the training and testing sets
        training_cpls = np.argwhere(self.labels_train > 0)
        testing_cpls = np.argwhere(self.labels_test > 0)

        # Filter these initial cpls so that samples do not go out of bounds
        # User could request a patch_size much larger than intended by DFC committee
        training_cpls = self._filter_cpls_by_image_edge_patch_size(training_cpls)
        testing_cpls = self._filter_cpls_by_image_edge_patch_size(testing_cpls)

        # Break the cpls out by class
        training_cpls_per_class = self._sort_cpls(training_cpls)
        testing_cpls_per_class = self._sort_cpls(testing_cpls)

        # Set the cpls
        self.training_cpls = training_cpls_per_class
        self.testing_cpls = testing_cpls_per_class
