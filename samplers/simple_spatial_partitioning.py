import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerSimpleSpatialPartitioning(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: False

    Article Bibtex Reference:
        @article{Friedl2000,
            abstract = {We present results from analyses conducted to evaluate the performance of advanced supervised classification 
            algorithms (decision trees and neural nets) applied to AVHRR data to map regional land cover in Central America. Our 
            results indicate that the sampling procedure used to stratify ground data into train and test sub-populations can 
            substantially bias accuracy assessment results. In particular, we found spatial autocorrelation in test data to inflate 
            estimates of classification accuracy by up to 50 points. Results from evaluations performed using independent train and 
            test data suggest that the feature space provided by AVHRR NDVI data is poorly suited for most land cover mapping problems, 
            with the exception of those involving highly generalized classes. © 2000 Taylor & Francis Group, LLC.},
            author = {M. A. Friedl and C. Woodcock and S. Gopal and D. Muchoney and A. H. Strahler and C. Barker-Schaaf},
            doi = {10.1080/014311600210434},
            issn = {13665901},
            issue = {5},
            journal = {International Journal of Remote Sensing},
            month = {1},
            pages = {1073-1077},
            publisher = {Taylor & Francis Group},
            title = {A note on procedures used for accuracy assessment in land cover maps derived from AVHRR data},
            volume = {21},
            url = {https://www.tandfonline.com/doi/abs/10.1080/014311600210434},
            year = {2000},
        }
    
    Article Sampler Description Excerpt:
    Section 2.2 "First, the data were stratifed into independent train and test sets using pixels selected at random from the entire 
    dataset, independent of the site from which they were derived. Second, the data were randomly stratifed into independent train and 
    test groups keeping data from individual sites together. We will refer to the former method as 'pixel-based splits' and the latter 
    as 'site-based splits'."

    Observations/Notes:
    - Friedl is the earliest identified reference to describe spatially partitioning data to ensure no overlap is present.
    - This implementation of spatial partitioning is simple (dumb) and could easily be improved with human input or other modifications.
    See SpatialPartitioning (not Simple). The only reason this method exists is to automate the process of spatial partitioning in the 
    simplest possible manner for empirical testing.
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

        # Find a split line simply by going train_ratio distance along the longest edge.
        split_line: Tuple[float, float] = (0.0, 0.0)  # y = mx + b
        longest_side_length = max(self.labels.shape)
        split_position = int(self.train_ratio * longest_side_length)
        if self.labels.shape[0] > self.labels.shape[1]:
            split_line = (0, split_position)
        else:
            split_line = (split_position, 0)

        # "Split" the cpls in each class as beloging to one side of the line or the other.
        positive_cpls = {}
        negative_cpls = {}
        for class_idx, cpls in valid_cpls_per_class.items():
            if len(cpls) < 1:
                continue  # Skip classes with no valid center pixel locations
            # Determine the side of the line each center pixel location belongs to
            side_per_cpl = np.sign(cpls[:, 0] - split_line[0] * cpls[:, 1] - split_line[1])
            # Assign side to each center pixel location
            positive_mask = side_per_cpl == 1
            negative_mask = side_per_cpl == -1
            positive_cpls[class_idx] = cpls[positive_mask]
            negative_cpls[class_idx] = cpls[negative_mask]
            # Ensure we remove any cpls that would result in a sample overlapping the split_line
            positive_cpls[class_idx] = self._filter_cpls_by_split_line(positive_cpls[class_idx], split_line)
            negative_cpls[class_idx] = self._filter_cpls_by_split_line(negative_cpls[class_idx], split_line)
        
        # Assign the positive/negative cpls as being training/testing using the split_ratio.
        total_positive_cpls = sum(len(cpls) for cpls in positive_cpls.values())
        total_negative_cpls = sum(len(cpls) for cpls in negative_cpls.values())

        if total_positive_cpls > total_negative_cpls:
            larger_set = positive_cpls
            smaller_set = negative_cpls
        else:
            larger_set = negative_cpls
            smaller_set = positive_cpls

        if self.train_ratio > 0.5:
            training_cpls_per_class = larger_set
            testing_cpls_per_class = smaller_set
        else:
            training_cpls_per_class = smaller_set
            testing_cpls_per_class = larger_set
        
        # Set the cpls
        self.training_cpls = training_cpls_per_class
        self.testing_cpls = testing_cpls_per_class
