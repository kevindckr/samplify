import os

# Ensure GeoPandas uses Shapely instead of PyGEOS
os.environ['USE_PYGEOS'] = '0'

import sys
import csv
import numba
import numpy as np
import scipy.stats
import pandas as pd
from typing import List, Union

from samplers._sampler_base import BaseImageSampler

import geopandas as gpd
from esda.moran import Moran
from libpysal.weights import Rook, DistanceBand


@numba.jit(nopython=True)
def generate_footprint_numba(image_y: int, image_x: int, cpls: np.ndarray, patch_bboxes: np.ndarray):
    """ 
    Given a set of cpls, calculate their footprint. Get a count per pixel location. 
    """
    footprint = np.zeros((image_y, image_x), dtype=np.int32)
    for y, x in cpls:
        ty, tx, by, bx = patch_bboxes[y, x]
        footprint[ty:by, tx:bx] += 1  # Increase the count of the pixel locations by 1
    return footprint


@numba.jit(nopython=True)
def count_overlapping_patches_numba(train_footprint: np.ndarray, test_cpls: np.ndarray, patch_bboxes: np.ndarray) -> int:
    """ 
    Count the number of test patches that overlap with any training patch.
    """
    overlapping_count = 0
    for idx in range(test_cpls.shape[0]):
        y, x = test_cpls[idx]
        ty, tx, by, bx = patch_bboxes[y, x]
        # Check if any pixel in the test patch overlaps with training patches
        if np.any(train_footprint[ty:by, tx:bx]):
            overlapping_count += 1
    return overlapping_count


class SamplerCharacterizer:
    
    def __init__(self, sampler: BaseImageSampler) -> None:
        self.sampler: BaseImageSampler = sampler

        # Overlap Percentage - "Mutually Exclusive Subset Assignment"
        self.overlap_percentage: Union[None, float] = None

        # Global Moran's I - "Spatial Autocorrelation"
        self.morans: Union[None, float] = None

        # KL Divergence - "Commensurate Class Distributions" (Ignore unlabeled class)
        self.kl_divergence_train: Union[None, float] = None
        self.kl_divergence_test: Union[None, float] = None

        # Different Ratio - "Bernoulli Distribution based Allocation"
        self.difference_ratio: Union[None, float] = None

        # ---- Other Information ----

        # Number of training and testing cpls
        self.num_train_cpls: int = 0
        self.num_test_cpls: int = 0

        # Number of available, used, and unused cpls
        self.num_initial_valid_cpls: int = 0
        self.num_used_cpls: int = 0
        self.num_unused_cpls: int = 0

        # Distribution of CPLs (classes, count)
        self.distribution_dataset: np.ndarray = None
        self.distribution_training: np.ndarray = None
        self.distribution_testing: np.ndarray = None

        # Indices of missing classes
        self.missing_training_classes: List[int] = []
        self.missing_testing_classes: List[int] = []

        # Number of cpls that are unlabelled
        self.num_unlabelled_train: int = 0
        self.num_unlabelled_test: int = 0

    def characterize_sampler(self):
        print(f'> Characterizing: {self.sampler.name} ({self.sampler.__class__.__name__})')
        self.calculate_overlap_ratio()
        self.calculate_morans()
        self.calculate_kldivergence()
        self.calculate_difference_ratio()

    def calculate_overlap_ratio(self):
        """ 
        Calculate the Overlap Percentage between training and testing samples as described in the article. 
        """
        # Retrieve relevant information from the sampler
        patch_bboxes = self.sampler.patch_bboxes
        image_height, image_width = self.sampler.image.shape[:2]
        test_cpls = np.concatenate(list(self.sampler.testing_cpls.values()), axis=0)
        train_cpls = np.concatenate(list(self.sampler.training_cpls.values()), axis=0)

        # Generate the footprint for training patches
        train_footprint = generate_footprint_numba(image_height, image_width, train_cpls, patch_bboxes)

        # Count the number of test patches that overlap with training patches
        overlapping_count = count_overlapping_patches_numba(train_footprint, test_cpls, patch_bboxes)

        # Total number of test patches
        total_test_patches = test_cpls.shape[0]

        # Calculate Overlap Percentage (OP)
        self.overlap_percentage = overlapping_count / total_test_patches

    def calculate_morans(self):
        """ Calculate Global Moran's I when encoding x's as "cpl in training set" (1) or "cpl in testing set" (0). """
        # Pull information we need from the sampler
        image_width, image_height = self.sampler.image.shape[0:2]
        test_cpls = np.concatenate(list(self.sampler.testing_cpls.values()), axis=0)
        train_cpls = np.concatenate(list(self.sampler.training_cpls.values()), axis=0)

        # Generate all possible locations in the image using numpy's meshgrid
        y_coords, x_coords = np.meshgrid(np.arange(image_width), np.arange(image_height), indexing='ij')
        all_locations = np.vstack([y_coords.ravel(), x_coords.ravel()]).T

        # Build the values provided to Morans (i.e. "x") based on encoding
        values = np.full((image_width, image_height), -1, dtype=int)
        values[test_cpls[:, 0], test_cpls[:, 1]] = 0
        values[train_cpls[:, 0], train_cpls[:, 1]] = 1

        # Flatten and filter the values to remove -1 (unused pixel location; not a cpl)
        values = values.ravel()
        valid_mask = values != -1
        filtered_values = values[valid_mask]
        filtered_locations = all_locations[valid_mask]

        # Convert cpl locations to GeoDataFrame and calculate Rook neighbor weight between them
        geom = gpd.points_from_xy(filtered_locations[:, 1], filtered_locations[:, 0])
        cpl_locations_df = gpd.GeoDataFrame(pd.DataFrame(filtered_locations, columns=['y', 'x']), geometry=geom)
        weights = Rook.from_dataframe(cpl_locations_df, use_index=False)

        # # Set a threshold distance (e.g., max distance between points)
        # threshold = np.mean(self.sampler.patch_size) * 2
        # weights = DistanceBand.from_dataframe(cpl_locations_df, threshold=threshold, binary=False)

        # Calculate Local Moran's I
        moran_result = Moran(filtered_values, weights)

        self.morans = moran_result.I
    
    def calculate_kldivergence(self):
        """ 
        Calculate KL Divergences between dataset-train and dataset-test ignoring unlabeled class. 
        
        Also calculates and stores other information that may be interesting.
        """
        # Calculate the distribution of the dataset 
        self.distribution_dataset = np.zeros(len(self.sampler.classes), dtype=np.int32)
        unique, counts = np.unique(self.sampler.labels, return_counts=True)
        unique = unique.astype(np.int32)  # Force back to int (return is Any)
        self.distribution_dataset[unique] = counts

        # Distribution of the training set
        self.distribution_training = np.zeros(len(self.sampler.classes), dtype=np.int32)
        for class_idx in self.sampler.classes.keys():
            if class_idx in self.sampler.training_cpls:
                self.distribution_training[class_idx] = len(self.sampler.training_cpls[class_idx])
            else:
                self.distribution_training[class_idx] = 0

        # Distribution of the testing set
        self.distribution_testing = np.zeros(len(self.sampler.classes), dtype=np.int32)
        for class_idx in self.sampler.classes.keys():
            if class_idx in self.sampler.testing_cpls:
                self.distribution_testing[class_idx] = len(self.sampler.testing_cpls[class_idx])
            else:
                self.distribution_testing[class_idx] = 0
        
        # Calculate what class indices may be missing in the training or testing (that are present in the dataset) (ignore unlabelled)
        self.missing_training_classes = [idx for idx in range(1, len(self.sampler.classes)-1) if self.distribution_dataset[idx] > 0 and self.distribution_training[idx] == 0]
        self.missing_testing_classes = [idx for idx in range(1, len(self.sampler.classes)-1) if self.distribution_dataset[idx] > 0 and self.distribution_testing[idx] == 0]

        # Calculate the number of unlabelled cpls in the training and testing sets
        self.num_unlabelled_train = int(self.distribution_training[0])
        self.num_unlabelled_test = int(self.distribution_testing[0])

        # Exclude the unlabelled class
        ignore_index = 0
        dataset_counts = np.delete(np.copy(self.distribution_dataset), ignore_index)
        training_counts = np.delete(np.copy(self.distribution_training), ignore_index)
        testing_counts = np.delete(np.copy(self.distribution_testing), ignore_index)

        # Convert counts to percentages
        total_count = np.sum(dataset_counts)
        total_train_count = np.sum(training_counts)
        total_test_count = np.sum(testing_counts)

        dataset_percents = dataset_counts / total_count
        train_percents = training_counts / total_train_count
        test_percents = testing_counts / total_test_count

        # Calculate KL-divergence
        self.kl_divergence_train = float(scipy.stats.entropy(train_percents, dataset_percents))
        self.kl_divergence_test = float(scipy.stats.entropy(test_percents, dataset_percents))

    def calculate_difference_ratio(self):
        """ 
        Calculates the Difference Ratio from desired and observed r_train. 
        
        Also calculates and stores other information that may be interesting.
        """
        # Get the number of training and testing cpls
        self.num_train_cpls = sum(len(cpls) for cpls in self.sampler.training_cpls.values())
        self.num_test_cpls = sum(len(cpls) for cpls in self.sampler.testing_cpls.values())

        # Get the number of used and unused cpls
        _, self.num_initial_valid_cpls = self.sampler._initial_cpls_per_class()
        self.num_used_cpls = self.num_train_cpls + self.num_test_cpls
        self.num_unused_cpls = self.num_initial_valid_cpls - self.num_used_cpls

        # Calculate the train to test ratio and its adherance to the selected value
        actual_train_ratio = self.num_train_cpls / self.num_used_cpls
        train_ratio_diff = abs(self.sampler.train_ratio - actual_train_ratio)

        # Calculate Difference Ratio
        self.difference_ratio = train_ratio_diff / self.sampler.train_ratio
