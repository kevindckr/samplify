import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

from sklearn.cluster import KMeans

from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerZhouControlled(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: True

    Article Bibtex Reference:
        @article{Zhou2015,
            abstract = {Joint spectral-spatial information based classification is an active topic in hyperspectral remote sensing. 
            Current classification approaches adopt a random sampling strategy to evaluate the performance of various classification 
            systems. Due to the limitation of benchmark data, sampling of training and testing data is performed on the same image. 
            In this paper, we point out that while training with random sampling is practical for hyperspectral image classification, 
            it has intrinsic problems in evaluating spectral-spatial information based classifiers. This statement is supported by 
            several experiments, and has lead to the proposal of a new sampling strategy for comparing spectral spatial information 
            based classifiers.},
            author = {Jun Zhou and Jie Liang and Yuntao Qian and Yongsheng Gao and Lei Tong},
            doi = {10.1109/WHISPERS.2015.8075474},
            isbn = {9781467390156},
            issn = {21586276},
            journal = {Workshop on Hyperspectral Image and Signal Processing, Evolution in Remote Sensing},
            keywords = {Hyperspectral classification,feature extraction,sampling,spectral-spatial analysis},
            month = {7},
            publisher = {IEEE Computer Society},
            title = {On the sampling strategies for evaluation of joint spectral-spatial information based classifiers},
            volume = {2015-June},
            year = {2015},
        }
    
    Article Sampler Description Excerpt:
    Section 4 "The new strategy shall avoid drawing samples homogeneously over the whole image and overlap between training and 
    testing regions. A direct approach is to sample continuously from a local area for each class. The randomness can be guaranteed 
    by choosing different local areas across the data. Though this approach can not completely eliminate overlap, the influence of testing 
    data on the training step can be greatly reduced"

    Observations/Notes:
    - There is very little specific information about the sampler's implementation other than the localization of training samples.
    - In this implementation local training sample selection areas are produced by utilizing the 2-clustering approach of Hansch. See
    SamplerHanschCluster observations and notes.
    - A stratified sampling of the total possible training CPLs from the 2-clusters is then performed to select from the local training 
    area. This is not a strict stratification if the train cluster cannot supply enough CPLs to match the stratification ratio over the 
    total training CPLs.
    - Overlap is allowed as the original article states.
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

        # Calculate all connected components (partitions) for all classes
        print(f'  Calculating all label partitions...')
        partitions_per_class = {}
        for class_idx, class_cpls in valid_cpls_per_class.items():
            partitions_per_class[class_idx] = self.find_connected_components(class_cpls)

        # For each class, for each partition grow a connected training region
        print(f'  Growing training regions in partitions...')
        train_cpls_per_class = {k: np.empty((0, 2), dtype=int) for k in self.classes.keys()}
        test_cpls_per_class = {k: np.empty((0, 2), dtype=int) for k in self.classes.keys()}

        for class_idx, class_partitions in partitions_per_class.items():
            # Zhou notes "from a local area for each class", so only chose A single partition for training
            partition_index = np.random.randint(len(class_partitions))
            training_partition = class_partitions[partition_index]
            testing_partitions = class_partitions[:partition_index] + class_partitions[partition_index + 1:]

            # Determine the target count of pixels in the region to form stratified distribution
            total_count = valid_cpls_per_class[class_idx].shape[0]
            train_count = int(self.train_ratio * total_count)

            # Grow a region of that size in the partition
            train_region, test_region, _ = self.grow_region(training_partition, region_ratio=None, target_region_size=train_count)

            # Place all selected CPLs into training set, rest into testing
            if train_region.size > 0:
                train_cpls_per_class[class_idx] = np.vstack([train_cpls_per_class[class_idx], train_region])
            if test_region.size > 0:
                test_cpls_per_class[class_idx] = np.vstack([test_cpls_per_class[class_idx], test_region])
            
            # Place all other partitions for the class into the test set
            for partition in testing_partitions:
                test_cpls_per_class[class_idx] = np.vstack([test_cpls_per_class[class_idx], partition])

        # Set the CPLs
        self.training_cpls = train_cpls_per_class
        self.testing_cpls = test_cpls_per_class






        # # Get the initial set of valid center pixel locations for each class
        # valid_cpls_per_class, total_valid_cpls = self._initial_cpls_per_class()

        # # Cluster each set of cpls per class into two clusters and assign clusters as train or test
        # init_train_cpls_per_class = {}
        # test_cpls_per_class = {}
        # for class_idx, cpls in valid_cpls_per_class.items():
        #     # Cluster each set of cpls per class into two clusters
        #     if len(cpls) < 2: continue 
        #     kmeans = KMeans(n_clusters=2, random_state=self.random_seed, n_init=10)
        #     kmeans.fit(cpls)
        #     labels = kmeans.labels_
        #     cluster0_cpls = cpls[labels == 0]
        #     cluster1_cpls = cpls[labels == 1]

        #     init_train_cpls_per_class[class_idx] = cluster0_cpls
        #     test_cpls_per_class[class_idx] = cluster1_cpls
        
        # total_train_cpls = sum(len(cpls) for cpls in init_train_cpls_per_class.values())
        
        # # Perform a stratified sampling of the initial training cpls per class
        # train_cpls_per_class = {}
        # for class_idx, class_cpls in init_train_cpls_per_class.items():
        #     # Get the ratio of this class to the entire valid set of CPLs
        #     class_ratio = valid_cpls_per_class[class_idx].shape[0] / total_valid_cpls
        #     # Of the total possible train CPLs, how many to satisfy class_ratio of the entire initially valid
        #     class_train_count = int(class_ratio * total_train_cpls)
        #     # Of the total stratified for this class, select train_ratio of them
        #     class_train_count = int(class_train_count * self.train_ratio)

        #     # Break stratification if we need more train CPLs to satsify than available
        #     class_count = class_cpls.shape[0]
        #     if class_train_count > class_count:
        #         train_cpls_per_class[class_idx] = class_cpls
        #     # Otherwise select a subset of the available
        #     elif class_train_count > 0:
        #         train_inds = np.random.choice(class_count, size=class_train_count, replace=False)
        #         train_cpls_per_class[class_idx] = class_cpls[train_inds]

        # # Set the cpls
        # self.training_cpls = train_cpls_per_class
        # self.testing_cpls = test_cpls_per_class
