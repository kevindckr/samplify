import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

from sklearn.cluster import KMeans

from samplers._overlap import cpl_validity_numba
from samplers._sampler_base import BaseImageSampler, numpy_deterministic


# Limit OpenBLAS (called during kmeans) to use a maximum of 4 threads
os.environ['OPENBLAS_NUM_THREADS'] = '4'

class SamplerHanschCluster(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: False

    Article Bibtex Reference:
        @article{Hansch2017,
            abstract = {The automatic generation of semantic maps from remotely sensed imagery by supervised classifiers has 
            seen much effort in the last decades. The major focus has been on the improvement of the interplay between feature 
            operators and classifiers, while experimental design and test data generation has been mostly neglected. This paper 
            shows that sampling strategies applied to partition the available reference data into train and test sets have 
            a large influence on the quality and reliability of the estimated generalization error. It illustrates and 
            discusses problems of common choices for sampling schemes, i.e. the violation of the independence assumption 
            and the illusion of the availability of global knowledge in the training data. Furthermore, a novel sampling 
            strategy is proposed which circumvents these problems and achieves a less biased estimate of the classification 
            error.},
            author = {R. Hansch and A. Ley and O. Hellwich},
            doi = {10.1109/IGARSS.2017.8127795},
            isbn = {9781509049516},
            journal = {International Geoscience and Remote Sensing Symposium (IGARSS)},
            keywords = {Error estimation,Sampling,Supervised classification},
            month = {12},
            pages = {3672-3675},
            publisher = {Institute of Electrical and Electronics Engineers Inc.},
            title = {Correct and still wrong: The relationship between sampling strategies and the estimation of the generalization error},
            volume = {2017-July},
            year = {2017},
        }
    
    Article Sampler Description Excerpt:
    Section Sampling Strategies "aims to mitigate all of the above mentioned problems by producing a balanced dataset with 
    minimal proximity (and thus minimal correlation) of train and test samples as well as a non-global distribution of the 
    training data. For each class the spatial coordinates of all samples are clustered into two clusters. Training samples
    of a class are randomly drawn from one of the clusters, the other cluster is used as test data. If two adjacent clusters 
    (of any classes) contribute to train and test data, a spatial border around the corresponding training samples ensures 
    non-overlapping train and test areas. In this way train and test samples of one class are maximally distinct from each 
    other (ensuring maximal independence between test and train samples) as well as being locally compact simulating the 
    application case where train and application areas are not in proximity."

    Observations/Notes:
    - It is unclear how the two clusters are assigned to be the traing or the testing cluster.
    - Empirical observation has found that regardless of how clusters are assigned each unique data set produces a unique ratio
    between the sizes of the resultant training and testing sets (after all class clusters have been assigned). For example, the
    Salinas data set has an approximately 0.5 ratio, the GRSS18 data set has an approximately 0.7 ratio.
    - As a result, the method can be data inefficient unless this "cluster ratio" is the same as the train_ratio. Otherwise many
    CPLs in the training set will be "wasted" when the train_ratio is lower than the "cluster ratio" IF the train_ratio is defined
    against the total count of valid CPLs. If it is defined against the total count of train CPLs the issue is even worse.
    - This issue cannot be fixed because we cannot place "wasted" train CPLs into the test set, it violates the method logic.
    - See also: "The Trap of Random Sampling and How to Avoid It: Alternative Sampling Strategies for a Realistic Estimate of 
    the Generalization Error in Remote Sensing" Hansch 2021.
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

        # Cluster each set of cpls per class into two clusters and assign clusters as train or test
        init_train_cpls_per_class = {}
        test_cpls_per_class = {}
        train_count, test_count = 0, 0
        for class_idx, cpls in valid_cpls_per_class.items():
            # Cluster each set of cpls per class into two clusters
            if len(cpls) < 2: continue 
            kmeans = KMeans(n_clusters=2, random_state=self.random_seed, n_init=10)
            kmeans.fit(cpls)
            labels = kmeans.labels_
            cluster0_cpls = cpls[labels == 0]
            cluster1_cpls = cpls[labels == 1]

            # Determine which is the larger and smaller cluster
            cluster0_count, cluster1_count = len(cluster0_cpls), len(cluster1_cpls)
            bigger_count, smaller_count = cluster0_count, cluster1_count
            bigger_set_cpls_per_class, smaller_set_cpls_per_class = cluster0_cpls, cluster1_cpls
            if cluster0_count > cluster1_count:
                smaller_count, bigger_count = cluster0_count, cluster1_count
                smaller_set_cpls_per_class, bigger_set_cpls_per_class = cluster0_cpls, cluster1_cpls

            # Determine what the current ratio between the train and test sets is
            target_ratio = 0.5  # self.train_ratio
            total_count = train_count + test_count
            current_ratio = target_ratio if total_count == 0 else train_count / (total_count)

            # Attempt to balance the number of cpls between the train and test sets. 
            #   This is seemingly a fruitless effort as a result of the 2-clustering in a stratified manner.
            #   Uncomment this print statement and watch the ratio over time and executions to see this.
            #     print(f'  Class: {class_idx} Ratio: {current_ratio}')
            if current_ratio <= target_ratio:
                init_train_cpls_per_class[class_idx] = bigger_set_cpls_per_class
                test_cpls_per_class[class_idx] = smaller_set_cpls_per_class
                train_count, test_count = train_count+bigger_count, test_count+smaller_count
            else:
                init_train_cpls_per_class[class_idx] = smaller_set_cpls_per_class
                test_cpls_per_class[class_idx] = bigger_set_cpls_per_class
                train_count, test_count = train_count+smaller_count, test_count+bigger_count
        
        # Subselect train_ratio number of train_cpls from the current set
        train_cpls_per_class = {}
        for class_idx, class_cpls in init_train_cpls_per_class.items():
            cpl_count = int(self.train_ratio * len(class_cpls))
            if cpl_count > 0:
                train_inds = np.random.choice(len(class_cpls), size=cpl_count, replace=False)
                train_cpls_per_class[class_idx] = class_cpls[train_inds]

        # Calculate the valid cpls in relation to the selected training cpls
        flat_train_cpls = np.concatenate(list(train_cpls_per_class.values()), axis=0)
        flat_test_cpls = np.concatenate(list(test_cpls_per_class.values()), axis=0)
        _, _, _, _, _, valid_test_cpls, _ = cpl_validity_numba(self.image.shape, flat_train_cpls, flat_test_cpls, self.patch_bboxes)

        # Set and Sort the cpls back to per class
        test_cpls = valid_test_cpls
        test_cpls_per_class = self._sort_cpls(test_cpls)

        # Set the cpls
        self.training_cpls = train_cpls_per_class
        self.testing_cpls = test_cpls_per_class
