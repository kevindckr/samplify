import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerRandomUniform(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: True

    No known or identified original reference. 

    Observations/Notes:
    - Generates incredible amounts of overlap.
    - Can only decrease overlap by having a larger train_ratio and only using small percentage of the set aside train CPLs.
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
        
        # Flatten the cpls from dict to single array for random split
        cpls = np.concatenate(list(valid_cpls_per_class.values()), axis=0)

        # Split, sort by class, and set the cpls for the training and testing sets
        self._randomsplit_sort_set_cpls(cpls)
