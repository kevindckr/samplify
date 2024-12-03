import os
import sys
import math
import numpy as np
from typing import Dict, Tuple, Optional

from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerRandomEqualCount(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: True

    No known or identified original reference. 

    Observations/Notes:
    - Generates and almost guarentees incredible amounts of overlap between train and test sets.
    - Can only decrease overlap by having a larger train_ratio and only using small percentage of the set aside train CPLs.
    """
    def __init__(self,
                 image: np.ndarray, labels: np.ndarray, classes: Dict[int, str],
                 patch_size: Tuple[int, int], train_ratio: float, 
                 name: str, random_seed: Optional[int] = None, **kwargs) -> None:
        super().__init__(image, labels, classes, patch_size, train_ratio, name, random_seed, **kwargs)

        # Default training_class_sample_count to 100 if not specified
        # TODO/NOTE: This is overridden below. This statement has no effect.
        self.training_class_sample_count = kwargs.get('training_class_sample_count', 100)

    @numpy_deterministic
    def sample(self) -> None:
        print(f'> Executing: {self.name} ({self.__class__.__name__})')

        # Get the initial set of valid center pixel locations for each class
        valid_cpls_per_class, total_count = self._initial_cpls_per_class()

        # Calculate the number of samples per class based on number of total valid cpls.
        train_cpl_count = (total_count * self.train_ratio)
        self.training_class_sample_count = int(train_cpl_count / (len(self.classes) - 1))  # Remove unlabelled

        # Randomly select training_class_sample_count cpls and rest to testing
        training_cpls_per_class = {k: np.empty((0, 2), dtype=int) for k in self.classes.keys()}
        testing_cpls_per_class = {k: np.empty((0, 2), dtype=int) for k in self.classes.keys()}
        for class_ind, class_cpls in valid_cpls_per_class.items():
            total_count = class_cpls.shape[0]
            if total_count == 0:
                continue
            # If we can satisfy the training_class_sample_count for this class, perform equal count split
            elif total_count > self.training_class_sample_count:
                train_inds = np.random.choice(total_count, size=self.training_class_sample_count, replace=False)
                test_inds = np.setdiff1d(np.arange(total_count), train_inds)

                training_cpls_per_class[class_ind] = class_cpls[train_inds]
                testing_cpls_per_class[class_ind] = class_cpls[test_inds]
            # If we cannot satisfy the training_class_sample_count, perform stratified split of this class
            else:
                train_count = int(self.train_ratio * total_count)
                train_inds = np.random.choice(total_count, size=train_count, replace=False)
                test_inds = np.setdiff1d(np.arange(total_count), train_inds)

                training_cpls_per_class[class_ind] = class_cpls[train_inds]
                testing_cpls_per_class[class_ind] = class_cpls[test_inds]

        # Set the cpls
        self.training_cpls = training_cpls_per_class
        self.testing_cpls = testing_cpls_per_class
