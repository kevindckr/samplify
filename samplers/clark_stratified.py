import os
import sys
import math
import numpy as np
from typing import Dict, Tuple, Optional

from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerClarkStratified(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: True

    Article Bibtex Reference:
        @article{Clark2023,
            abstract = {Convolutional Neural Networks (CNN) consist of various hyper-parameters which need to be specified 
            or can be altered when defining a deep learning architecture. There are numerous studies which have tested 
            different types of networks (e.g. U-Net, DeepLabv3+) or created new architectures, benchmarked against 
            well-known test datasets. However, there is a lack of real-world mapping applications demonstrating the 
            effects of changing network hyper-parameters on model performance for land use and land cover (LULC) semantic 
            segmentation. In this paper, we analysed the effects on training time and classification accuracy by altering 
            parameters such as the number of initial convolutional filters, kernel size, network depth, kernel initialiser 
            and activation functions, loss and loss optimiser functions, and learning rate. We achieved this using a 
            well-known top performing architecture, the U-Net, in conjunction with LULC training data and two 
            multispectral aerial images from North Queensland, Australia. A 2018 image was used to train and test CNN 
            models with different parameters and a 2015 image was used for assessing the optimised parameters. We found 
            more complex models with a larger number of filters and larger kernel size produce classifications of higher 
            accuracy but take longer to train. Using an accuracy-time ranking formula, we found using 56 initial filters 
            with kernel size of 5x5 provide the best compromise between training time and accuracy. When fully training 
            a model using these parameters and testing on the 2015 image, we achieved a kappa score of 0.84. This compares 
            to the original U-Net parameters which achieved a kappa score of 0.73.},
            author = {Andrew Clark and Stuart Phinn and Peter Scarth},
            doi = {10.1007/S41064-023-00233-3/FIGURES/17},
            issn = {25122819},
            issue = {2},
            journal = {PFG - Journal of Photogrammetry, Remote Sensing and Geoinformation Science},
            keywords = {Aerial imagery,Convolutional neural network,Deep learning,Land cover,Land use,Semantic segmentation},
            month = {4},
            pages = {125-147},
            publisher = {Springer Science and Business Media Deutschland GmbH},
            title = {Optimised U-Net for Land Use-Land Cover Classification Using Aerial Photography},
            volume = {91},
            url = {https://link-springer-com.afit.idm.oclc.org/article/10.1007/s41064-023-00233-3},
            year = {2023},
        }
    
    Article Sampler Description Excerpt:
    Section 2.4 "we calculated the number of patches per class by multiplying the total number of patches by the log of the 
    class area divided by the sum of the log of the project area. The result was rounded up to the nearest integer (Eq. 1)
    where N cp is the number of class patches, N p is the total number of required training patches, and a c is the class area.
    For each feature, we distributed the number of class patches based on the proportion of the area that the feature 
    represents rounded up to the nearest integer (Eq. 2) where Nfp is the number of feature patches, Ncp is the number of 
    class patches, af is the feature area, and ac is the class area. Once we calculated the number of patches per feature, we 
    randomly generated patch locations ensuring that the centre of the patch was located within the feature.

    Observations/Notes:
    - It is unclear what is exactly meant by, and the interrerlation of, the phrases "total number of patches", "class area", 
    "project area", "feature patch", "class patch".
    - In this implementation our best judgement is that the "class area" is the number of labelled pixels the class contains, the
    "project area" is the total number of labelled pixels, and "feature" is another word for class/label. Thus, this implementation
    finds the total area of each class, the total area of the labels overall, and assigns the ratio of the log of the class area 
    over the log of the total area.
    - This method effectively better represents under-represented classes. 
    - This method still can, and does, result in large amounts overlap between the training and testing sets.
    """
    def __init__(self,
                 image: np.ndarray, labels: np.ndarray, classes: Dict[int, str],
                 patch_size: Tuple[int, int], train_ratio: float, 
                 name: str, random_seed: Optional[int] = None, **kwargs) -> None:
        super().__init__(image, labels, classes, patch_size, train_ratio, name, random_seed, **kwargs)

    @numpy_deterministic
    def sample(self) -> None:
        print(f'> Executing: {self.name} ({self.__class__.__name__})')

        # Get the initial set of valid center pixel locations for each class
        valid_cpls_per_class, _ = self._initial_cpls_per_class()

        # Assume the "class area" is the number of valid_cpls it has
        class_areas = {k: len(cpls) for k, cpls in valid_cpls_per_class.items()}
        total_cpl_count = sum(class_areas.values())

        # Calculate the number of patches per class based on the log_area_total
        log_area_total = sum(math.log(area) for area in class_areas.values())
        cpls_per_class = {class_idx: math.ceil((math.log(area) / log_area_total) * total_cpl_count)
                             for class_idx, area in class_areas.items()}

        # Do stratified selection based on the cpls_per_class
        training_cpls_per_class = {k: np.empty((0, 2), dtype=int) for k in self.classes.keys()}
        testing_cpls_per_class = {k: np.empty((0, 2), dtype=int) for k in self.classes.keys()}
        for class_idx, class_cpls in valid_cpls_per_class.items():
            cpl_count = cpls_per_class[class_idx]
            cpl_count = min(cpl_count, len(class_cpls))  # Don't overrun
            if cpl_count > 0:
                # Randomly select cpls_per_class cpls from the valid set
                class_cpls_inds = np.random.choice(len(class_cpls), size=cpl_count, replace=False)
                class_cpls = class_cpls[class_cpls_inds]
                
                # Perform a stratified split based on the train ratio of these CPLs
                train_count = int(self.train_ratio * len(class_cpls))
                train_inds = np.random.choice(len(class_cpls), size=train_count, replace=False)
                test_inds = np.setdiff1d(np.arange(len(class_cpls)), train_inds)
                training_cpls_per_class[class_idx] = class_cpls[train_inds]
                testing_cpls_per_class[class_idx] = class_cpls[test_inds]
        
        # Set the cpls
        self.training_cpls = training_cpls_per_class
        self.testing_cpls = testing_cpls_per_class
