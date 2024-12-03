import os
import sys
import math
import numpy as np
from typing import Dict, Tuple, Optional

from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerZouGrid(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: False

    Article Bibtex Reference:
        @article{Zou2020,
            abstract = {Due to its remarkable feature representation capability and high performance, convolutional neural networks 
            (CNN) have emerged as a popular choice for hyperspectral image (HSI) analysis. However, the performances of traditional 
            CNN-based patch-wise classification methods are limited by insufficient training samples, and the evaluation strategies 
            tend to provide overoptimistic results due to training-test information leakage. To address these concerns, we propose 
            a novel spectral-spatial 3-D fully convolutional network (SS3FCN) to jointly explore the spectral-spatial information 
            and the semantic information. SS3FCN takes small patches of original HSI as inputs and produces the corresponding sized 
            outputs, which enhances the utilization rate of the scarce labeled images and boosts the classification accuracy. In 
            addition, to avoid the potential information leakage and make a fair comparison, we introduce a new principle to 
            generate classification benchmarks. Experimental results on four popular benchmark datasets, including Salinas Valley, 
            Pavia University, Indian Pines, and Houston University, demonstrate that the SS3FCN outperforms state-of-the-art methods 
            and can be served as a baseline for future research on HSI classification.},
            author = {Liang Zou and Xingliang Zhu and Changfeng Wu and Yong Liu and Lei Qu},
            doi = {10.1109/JSTARS.2020.2968179},
            issn = {21511535},
            journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
            keywords = {3-D fully convolutional networks,Hyperspectral image (HSI) classification,spectral-spatial exploration},
            pages = {659-674},
            publisher = {Institute of Electrical and Electronics Engineers},
            title = {Spectral-Spatial Exploration for Hyperspectral Image Classification via the Fusion of Fully Convolutional Networks},
            volume = {13},
            year = {2020},
        }
    
    Article Sampler Description Excerpt:
    Section II.A "First, we divide the whole HSI into blocks with the size of WxHxC, where W and H are the width and height, C is 
    the number of spectral bands" ... "discard the blocks where all the pixels are unlabeled and select the blocks with only one kind of 
    pixels as the test set, denoted as test set-1. The remaining blocks with more than one class of pixels are sorted column-wisely 
    following their orders in the HS" ... "all the multiclass blocks are partitioned into K folds, where the subsequent two blocks in each 
    fold is K apart in term of the order. The parameter K is limited by the percentage of pixels taken as training samples" ... "We select 
    a single fold as the training set, the other one as the validation set, and the remaining seven folds as the test set-2"

    Observations/Notes:
    - It is unclear what the description about the limiting value of K means. It is assumed it just means that K cannot be larger than the
    number of multiclass windows.
    """
    def __init__(self,
                 image: np.ndarray, labels: np.ndarray, classes: Dict[int, str],
                 patch_size: Tuple[int, int], train_ratio: float, 
                 name: str, random_seed: Optional[int] = None, **kwargs) -> None:
        super().__init__(image, labels, classes, patch_size, train_ratio, name, random_seed, **kwargs)

        # Default multiclass_window_folds (K) to 9 (default from article) if not provided
        self.multiclass_window_folds: Tuple[int, int] = kwargs.get('multiclass_window_folds', 9)

    @numpy_deterministic
    def sample(self) -> None:
        print(f'> Executing: {self.name} ({self.__class__.__name__})')

        # Calculate the number of samples that fit into the image given the patch_size
        num_slices_x = math.floor(((self.image.shape[1] - self.patch_size[1]) / self.patch_size[1]) + 1)
        num_slices_y = math.floor(((self.image.shape[0] - self.patch_size[0]) / self.patch_size[0]) + 1)

        # Collect the center pixel locations for samples in the grid
        multiclass_windows = []
        multiclass_class_counts = np.zeros(len(self.classes))
        testing_cpls_per_class = {k: [] for k in self.classes.keys()}
        for y in range(num_slices_y):
            for x in range(num_slices_x):
                # Calculate the CPL for this slice
                center_x = x * self.patch_size[1] + self.patch_size[1] // 2
                center_y = y * self.patch_size[0] + self.patch_size[0] // 2

                # Get the extends of this slice
                ty, tx, by, bx = self.patch_bboxes[center_y, center_x] 

                # Slice the labels and determine what type of slice this is
                label_slice = self.labels[ty:by, tx:bx]
                cpl_class = self.labels[center_y, center_x]

                # Sort the slice by its type
                if np.all(label_slice == 0):
                    # Entirely unlabelled slice
                    pass
                elif np.all(label_slice == cpl_class):
                    # Entirely cpl_class slice (Guarenteed not unlabelled)
                    testing_cpls_per_class[cpl_class].append((center_y, center_x))
                else:
                    # Calculate which classes occupy this slice
                    unique = np.unique(label_slice)

                    # Create an array representing the occupancy to store with the cpl
                    class_occupancy = np.zeros(len(self.classes))
                    class_occupancy[unique] = 1

                    # Store a global count across all multiclass windows of the class counts
                    multiclass_class_counts[unique] += 1

                    # Multiclass window (potentially unalabelled cpl)
                    multiclass_windows.append((center_y, center_x, cpl_class, class_occupancy))

        # Split multiclass windows into train and test by making k folds
        k = self.multiclass_window_folds
        fold_size = math.ceil(len(multiclass_windows) / k)
        training_cpls_per_class = {k: [] for k in self.classes.keys()}
        for fold_idx in range(fold_size):
            # Calculate the fold indices
            start = fold_idx * k
            end = min(len(multiclass_windows), (fold_idx + 1) * k)
            windows = multiclass_windows[start:end]
            if len(windows) > 0:
                # Assign each window in the fold to a specific set
                for window_ind in range(len(windows)):
                    y, x, cpl_class, _ = windows[window_ind]
                    # One value goes to training
                    if window_ind == 0:
                        training_cpls_per_class[cpl_class].append((y, x))
                    # The rest go to test
                    else:
                        testing_cpls_per_class[cpl_class].append((y, x))
                    # If we support val set, we would add one (window_ind == 1) to val set also
        
        # Convert from Dict[int, List[Tuple]] to Dict[int, np.ndarray]
        converted_training_cpls_per_class = {}
        for class_idx, class_cpls in training_cpls_per_class.items():
            if len(class_cpls) == 0: continue
            converted_training_cpls_per_class[class_idx] = np.row_stack(class_cpls)
        
        converted_testing_cpls_per_class = {}
        for class_idx, class_cpls in testing_cpls_per_class.items():
            if len(class_cpls) == 0: continue
            converted_testing_cpls_per_class[class_idx] = np.row_stack(class_cpls)

        # Set the cpls
        self.training_cpls = converted_training_cpls_per_class
        self.testing_cpls = converted_testing_cpls_per_class
