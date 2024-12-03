import os
import sys
import math
import numpy as np
from typing import Dict, Tuple, Optional

from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerDeckerDynamic(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: False

    Article Reference: "A Survey of Sampling Methods for Hyperspectral Remote Sensing: Addressing Bias Induced by Random Sampling"
    ** This algorithm never made it into the article **

    Observations/Notes:
    - This method adheres to train_ratio moderately.
    - Very sensitive to grid_size parameter!
    - Poor train-test ratio adherence.
    """
    def __init__(self,
                 image: np.ndarray, labels: np.ndarray, classes: Dict[int, str],
                 patch_size: Tuple[int, int], train_ratio: float, 
                 name: str, random_seed: Optional[int] = None, **kwargs) -> None:
        super().__init__(image, labels, classes, patch_size, train_ratio, name, random_seed, **kwargs)

        # Default grid_size to 4 * patch_size
        default_grid_size = tuple(int(t * 3) for t in self.patch_size)
        self.grid_size: Tuple[int, int] = kwargs.get('grid_size', default_grid_size)

    @numpy_deterministic
    def sample(self) -> None:
        print(f'> Executing: {self.name} ({self.__class__.__name__})')

        # Calculate the grid cells in each direction
        num_slices_y = math.floor(self.image.shape[0] / self.grid_size[0])
        num_slices_x = math.floor(self.image.shape[1] / self.grid_size[1])
        num_grid_cells = num_slices_y * num_slices_x

        # Create the cell designations, 0 for training, 1 for testing.
        train_cells = int(num_grid_cells * self.train_ratio)
        cell_designations = np.ones(num_grid_cells)
        cell_designations[:train_cells] = 0

        # For each grid cell index, calculate its boundaries
        cell_boundaries = np.zeros((num_grid_cells, 4), dtype=np.int32)
        for y in range(num_slices_y):
            for x in range(num_slices_x):
                cell_idx = y * num_slices_x + x
                min_y, min_x = y * self.grid_size[0], x * self.grid_size[1]
                max_y, max_x = min_y + self.grid_size[0], min_x + self.grid_size[1]

                # Combine fractional grid_size cells together at the ends of rows and columns
                if (min_y + 2 * self.grid_size[0]) > self.image.shape[0]:
                    max_y = self.image.shape[0]
                if (min_x + 2 * self.grid_size[1]) > self.image.shape[1]:
                    max_x = self.image.shape[1]

                cell_boundaries[cell_idx] = [min_y, min_x, max_y, max_x]

        # Get the initial set of valid center pixel locations for each class
        valid_cpls_per_class, _ = self._initial_cpls_per_class()
        valid_cpls = np.concatenate(list(valid_cpls_per_class.values()), axis=0)

        # Create an image of the initially valid cpls for the train and test set
        valid_train_image = np.zeros_like(self.labels)
        valid_test_image = np.zeros_like(self.labels)
        valid_train_image[valid_cpls[:, 0], valid_cpls[:, 1]] = 1
        valid_test_image[valid_cpls[:, 0], valid_cpls[:, 1]] = 1

        # Continually sample until no valid locations are left
        train_cpls_per_class = {k: [] for k in self.classes.keys()}
        test_cpls_per_class = {k: [] for k in self.classes.keys()}
        while np.any(valid_train_image == 1) or np.any(valid_test_image == 1):
            # Randomize the grid's cell designations
            np.random.shuffle(cell_designations)

            # Select a new sample in each cell
            for cell_idx, cell_designation in enumerate(cell_designations):
                # Get the cell boundaries for this cell
                min_y, min_x, max_y, max_x = cell_boundaries[cell_idx]

                # Get the current set of pixels for this designation type
                valid_image = valid_train_image if cell_designation == 0 else valid_test_image

                # Get the pixels within this cell for the training set
                cell_pixels = valid_image[min_y:max_y, min_x:max_x]

                # Get the valid cpls in this cell, if empty, no samples to draw
                valid_cpls = np.where(cell_pixels == 1)
                if len(valid_cpls[0]) == 0:
                    continue
                
                # Randomly promote one of the cpls to be selected
                selected_idx = np.random.choice(len(valid_cpls[0]))
                cpl_y, cpl_x = valid_cpls[0][selected_idx], valid_cpls[1][selected_idx]
                cpl_y, cpl_x = cpl_y + min_y, cpl_x + min_x  # Offset back to image coordiantes

                # Get the overlap and patch boundaries of this cpl
                cpl_class = self.labels[cpl_y, cpl_x]
                over_min_y, over_min_x, over_max_y, over_max_x = self.over_bboxes[cpl_y, cpl_x]

                if cell_designation == 0:
                    # Add this selected cpl to the training set
                    train_cpls_per_class[cpl_class].append((cpl_y, cpl_x))
                    # Remove testing cpls that would fall within the overlap boundary of the selected train cpl
                    valid_test_image[over_min_y:over_max_y, over_min_x:over_max_x] = 0
                    # Remove this training cpl from the set of valid training cpls for this cell (allow training patch overlap)
                    valid_train_image[cpl_y, cpl_x] = 0
                else:
                    # Add this selected cpl to the training set
                    test_cpls_per_class[cpl_class].append((cpl_y, cpl_x))
                    # Remove testing cpls that would fall within the overlap boundary of the selected test cpl
                    valid_train_image[over_min_y:over_max_y, over_min_x:over_max_x] = 0
                    # Remove this testing cpl from the set of valid testing cpls for this cell (allow testing patch overlap)
                    valid_test_image[cpl_y, cpl_x] = 0

        # Convert the lists to np.ndarray
        converted_training_cpls_per_class = {}
        for class_idx, class_cpls in train_cpls_per_class.items():
            if len(class_cpls) == 0: continue
            converted_training_cpls_per_class[class_idx] = np.row_stack(class_cpls)
        
        converted_testing_cpls_per_class = {}
        for class_idx, class_cpls in test_cpls_per_class.items():
            if len(class_cpls) == 0: continue
            converted_testing_cpls_per_class[class_idx] = np.row_stack(class_cpls)

        # Set the cpls
        self.training_cpls = converted_training_cpls_per_class
        self.testing_cpls = converted_testing_cpls_per_class
