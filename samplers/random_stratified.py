import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerRandomStratified(BaseImageSampler):
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

        # Stratified random split the cpls for each class into training and testing
        training_cpls_per_class = {k: np.empty((0, 2), dtype=int) for k in self.classes.keys()}
        testing_cpls_per_class = {k: np.empty((0, 2), dtype=int) for k in self.classes.keys()}
        for class_ind, class_cpls in valid_cpls_per_class.items():
            total_count = class_cpls.shape[0]
            if total_count > 0:
                train_count = int(self.train_ratio * total_count)
                train_inds = np.random.choice(total_count, size=train_count, replace=False)
                test_inds = np.setdiff1d(np.arange(total_count), train_inds)

                training_cpls_per_class[class_ind] = class_cpls[train_inds]
                testing_cpls_per_class[class_ind] = class_cpls[test_inds]
        
        # Set the cpls
        self.training_cpls = training_cpls_per_class
        self.testing_cpls = testing_cpls_per_class
