import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from samplers._sampler_base import BaseImageSampler, numpy_deterministic


# Limit OpenBLAS (called during kmeans) to use a maximum of 4 threads
os.environ['OPENBLAS_NUM_THREADS'] = '4'

class SamplerDeckerSpatial(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: False

    Article Reference: "A Survey of Sampling Methods for Hyperspectral Remote Sensing: Addressing Bias Induced by Random Sampling"
    ** The article renames this to Clustered Spatial **

    Observations/Notes:
    - This method cannot adhere to the split ratio, only very weakly.
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

        # Cluster each set of cpls per class into two clusters and obtain the centroid of each cluster.
        centroids_per_class = {}
        for class_idx, cpls in valid_cpls_per_class.items():
            if len(cpls) < 2:
                continue  # Skip classes with insufficient samples for clustering
            kmeans = KMeans(n_clusters=2, random_state=self.random_seed, n_init=10)
            kmeans.fit(cpls)
            centroids_per_class[class_idx] = kmeans.cluster_centers_

        # Find a global "clustering of clusters" to try to split by
        split_line: Tuple[float, float] = (0.0, 0.0)  # y = mx + b
        valid_centroids = np.vstack([centroids for centroids in centroids_per_class.values() if len(centroids) == 2])
        if len(valid_centroids) < 4:
            print(f'! Not enough valid centroids per class to cluster globally !')
            print(f'! Defaulting to orthogonal split line across largest edge  !')
            longest_side_length = max(self.labels.shape)
            split_position = int(self.train_ratio * longest_side_length)
            if self.labels.shape[0] > self.labels.shape[1]:
                split_line = (0, split_position)
            else:
                split_line = (split_position, 0)
        else:
            # Cluster on the class clusters.
            kmeans_global = KMeans(n_clusters=2, random_state=self.random_seed, n_init=10)
            kmeans_global.fit(valid_centroids)
            centroids_global = kmeans_global.cluster_centers_

            # Fit a line to separate the global centroids
            X_global = centroids_global[:, 1].reshape(-1, 1)
            y_global = centroids_global[:, 0]
            model = LinearRegression()
            model.fit(X_global, y_global)
            split_line = (model.coef_[0], model.intercept_)

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
