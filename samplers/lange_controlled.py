import os
import sys
import numpy as np
from collections import deque
from typing import Dict, Tuple, List, Optional

from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerLangeControlled(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: True

    Article Bibtex Reference:
        @article{Lange2018,
            abstract = {Supervised image classification is one of the essential techniques for generating semantic maps from 
            remotely sensed images. The lack of labeled ground truth datasets, due to the inherent time effort and cost involved 
            in collecting training samples, has led to the practice of training and validating new classifiers within a single 
            image. In line with that, the dominant approach for the division of the available ground truth into disjoint training 
            and test sets is random sampling. This paper discusses the problems that arise when this strategy is adopted in 
            conjunction with spectral-spatial and pixel-wise classifiers such as 3D Convolutional Neural Networks (3D CNN). 
            It is shown that a random sampling scheme leads to a violation of the independence assumption and to the illusion 
            that global knowledge is extracted from the training set. To tackle this issue, two improved sampling strategies 
            based on the Density-Based Clustering Algorithm (DBSCAN) are proposed. They minimize the violation of the train 
            and test samples independence assumption and thus ensure an honest estimation of the generalization capabilities 
            of the classifier.},
            author = {Julius Lange and Gabriele Cavallaro and Markus Götz and Ernir Erlingsson and Morris Riedel},
            doi = {10.1109/IGARSS.2018.8518671},
            isbn = {9781538671504},
            journal = {International Geoscience and Remote Sensing Symposium (IGARSS)},
            keywords = {Clustering,Convolutional Neural Networks (CNNs),DBSCAN,Deep learning,Hyperspectral image 
            classification,Sampling strategies},
            month = {10},
            pages = {2087-2090},
            publisher = {Institute of Electrical and Electronics Engineers Inc.},
            title = {The influence of sampling methods on pixel-wise hyperspectral image classification with 3D 
            convolutional neural networks},
            volume = {2018-July},
            year = {2018},
        }
    
    Article Sampler Description Excerpt:
    Section 3.1 "first, extracting larger contiguous regions using the class labels, e.g. buildings, fields, etc., and then 
    distributing these disjointly between the training and test set." ... "extraction of the contiguous regions is achieved 
    with the DBSCAN clustering algorithm. It detects subgroups within a set through the recursive evaluation of a neighbor 
    point density threshold (minPoints) criterion within a parametric search radius (ε) around a sample" ... "an approach 
    should be selected that maximizes the variability in the training set, so that a large number of potential patterns is 
    covered. This requires to establish a metric that evaluates said variety. The first two, proposed as part of this work, 
    are the region area size and statistical variance. Based on this, sorting the regions in ascending, respectively descending 
    order, and assigning them to the training set, up until the selected split percentage, should result in a less biased but 
    highly variable pattern distribution"

    Observations/Notes:
    - Like Liang, this article allows overlap between train and test samples and faces this issue by forming regions of
    contiguous CPLs that then "protect" inner CPLs from overlap.
    - The article states that it uses the Indian Pines data set and specifically the "forest" class. The common Indian Pines
    data set does not contain such a class. It is unclear what data set they used to develop and test their method(s).
    - What is meant by the variance of an image slice is not explicitly stated.
    - The results of the clustering of classes using DBSCAN will be dependent on the density and search radius parameters. The
    parameters used are not stated in the article and more than likely would need to be tuned for each data set. More than likely,
    it would also need to be tuned for each class so that larger regions of over represented classes do not generate numerous
    sub-regions.
    - The stated purpose of using DBSCAN is to extract contiguous regions of class labels. Though, based on how DBSCAN works and 
    Figure 1.c in the article, it results in sub-regions of an otherwise larger region. 
    - Given the potential issue with DBSCAN we calculate all connected components for each class and subplant these as regions.
    - Regardless of the clustering method, this can result in either inefficient data usage (see also SamplerHanschCluster) or
    something akin to random uniform sampling (when clusters are approximately the patch size)

    Changes from described implementation:
    - To ensure that at least one region for each class appears in the training set we iterate over classes and select highest 
    variance per class (at least once) until the train_ratio is fulfilled. This marginally overcomes issues with having to 
    "tune" DBScan for each dataset by hand. This is done for automation of experiments.
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
        valid_cpls_per_class, total_count = self._initial_cpls_per_class()

        # Calculate all connected components (partitions) for all classes
        partitions_per_class = {cls: self.find_connected_components(cpls) for cls, cpls in valid_cpls_per_class.items()}
        
        # Define what the variance of a slice of the image means
        def partition_average_spatial_variance(partition):
            flattened_spatial = self.image[partition[:, 0], partition[:, 1], :]
            spatial_variance_per_band = np.var(flattened_spatial, axis=0)
            average_spatial_variance = np.mean(spatial_variance_per_band)
            return average_spatial_variance
        
        # Sort all regions by variance
        partitions_sorted_by_variance_per_class = {
            cls: sorted(partitions, key=partition_average_spatial_variance, reverse=True)
            for cls, partitions in partitions_per_class.items()
        }

        # Assign partitions to the training set until the train_ratio is reached
        # Ensure all classes have been iterated at least once to ensure more balanced training set
        train_count = 0
        training_cpls = []
        iterated_all_classes_once = False
        train_ratio_reached = lambda: train_count / total_count > self.train_ratio
        while not train_ratio_reached():
            # Iterate over each class and add highest variance region to training
            for _, partitions in partitions_sorted_by_variance_per_class.items():
                if partitions:
                    partition = partitions.pop(0)
                    training_cpls.extend(partition)
                    train_count += len(partition)
                # If we already iterated all classes once, stop adding 
                if iterated_all_classes_once and train_ratio_reached():
                        break
            # We have now iterated all classes atleast once
            iterated_all_classes_once = True

        # Collect remaining partitions for the testing set
        testing_cpls = []
        for partitions in partitions_sorted_by_variance_per_class.values():
            for partition in partitions:
                testing_cpls.extend(partition)
        
        training_cpls = np.array(training_cpls)
        testing_cpls = np.array(testing_cpls)
        
        # Break the cpls out by class
        training_cpls_per_class = self._sort_cpls(training_cpls)
        testing_cpls_per_class = self._sort_cpls(testing_cpls)

        # Set the cpls
        self.training_cpls = training_cpls_per_class
        self.testing_cpls = testing_cpls_per_class
