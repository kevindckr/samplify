import os
import sys
import math
import numpy as np
from typing import Dict, Tuple, Optional

from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerGrid(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: False (No Stride), True (With Stide)

    No known or identified original reference. 

    Observations/Notes:
    - Removes grid cells entirely composed of an unlabelled class.
    """
    def __init__(self,
                 image: np.ndarray, labels: np.ndarray, classes: Dict[int, str],
                 patch_size: Tuple[int, int], train_ratio: float, 
                 name: str, random_seed: Optional[int] = None, **kwargs) -> None:
        super().__init__(image, labels, classes, patch_size, train_ratio, name, random_seed, **kwargs)

        # Default grid_stride to patch_size if not provided (that is, no stride)
        self.grid_stride: Tuple[int, int] = kwargs.get('grid_stride', patch_size)

    @numpy_deterministic
    def sample(self) -> None:
        print(f'> Executing: {self.name} ({self.__class__.__name__})')

        # Calculate the number of samples that fit into the image given the stride and patch_size
        num_slices_x = math.floor(((self.image.shape[1] - self.patch_size[1]) / self.grid_stride[1]) + 1)
        num_slices_y = math.floor(((self.image.shape[0] - self.patch_size[0]) / self.grid_stride[0]) + 1)

        # Collect the center pixel locations for samples in the grid
        valid_cpls = []
        for y in range(num_slices_y):
            for x in range(num_slices_x):
                # Calculate the CPL for this slice
                start_x = x * self.grid_stride[1]
                start_y = y * self.grid_stride[0]
                center_x = start_x + self.patch_size[1] // 2
                center_y = start_y + self.patch_size[0] // 2

                # Get the extends of this slice
                ty, tx, by, bx = self.patch_bboxes[center_y, center_x] 

                # Check if the slice is not entirely unlabelled
                label_slice = self.labels[ty:by, tx:bx]
                if not np.all(label_slice == 0):
                    valid_cpls.append((center_y, center_x))
        
        valid_cpls = np.array(valid_cpls)
        
        # Split, sort by class, and set the cpls for the training and testing sets
        self._randomsplit_sort_set_cpls(valid_cpls)
