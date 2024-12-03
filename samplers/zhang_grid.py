import os
import sys
import math
import numpy as np
from typing import Dict, Tuple, Optional

from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerZhangGrid(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: False

    Article Bibtex Reference:
        @article{Zhang2023b,
            abstract = {Transformer is a powerful tool for capturing long-range dependencies and has shown impressive performance 
            in hyperspectral image (HSI) classification. However, such power comes with a heavy memory footprint and huge 
            computation burden. In this article, we propose two types of lightweight self-attention modules (a channel lightweight
              multihead self-attention (CLMSA) module and a position lightweight multihead self-attention (PLMSA) module) to 
              reduce both memory and computation while associating each pixel or channel with global information. Moreover, we 
              discover that transformers are ineffective in explicitly extracting local and multiscale features due to the fixed 
              input size and tend to overfit when dealing with a small number of training samples. Therefore, a lightweight 
              transformer (LiT) network, built with the proposed lightweight self-attention modules, is presented. LiT adopts 
              convolutional blocks to explicitly extract local information in early layers and employs transformers to capture 
              long-range dependencies in deep layers. Furthermore, we design a controlled multiclass stratified (CMS) sampling 
              strategy to generate appropriately sized input data, ensure balanced sampling, and reduce the overlap of feature 
              extraction regions between training and test samples. With appropriate training data, convolutional tokenization, 
              and LiTs, LiT mitigates overfitting and enjoys both high computational efficiency and good performance. Experimental 
              results on several HSI datasets verify the effectiveness of our design.},
            author = {Xuming Zhang and Yuanchao Su and Lianru Gao and Lorenzo Bruzzone and Xingfa Gu and Qingjiu Tian},
            doi = {10.1109/TGRS.2023.3297858},
            issn = {15580644},
            journal = {IEEE Transactions on Geoscience and Remote Sensing},
            keywords = {Deep learning (DL),hyperspectral image (HSI) classification,transformer},
            publisher = {Institute of Electrical and Electronics Engineers Inc.},
            title = {A Lightweight Transformer Network for Hyperspectral Image Classification},
            volume = {61},
            year = {2023},
        }
    
    Article Sampler Description Excerpt:
    Section III.B "first partitions the entire HSI into nonoverlapping windows (Step 1). The window size should ensure that each class 
    exists in at least two windows to guarantee that all classes are present in both training and test sets. The CMS sampler will pad 
    the HSI when the height and width cannot be divided by window size. Windows where all pixels have the same labeled class are used as 
    test data, and windows where all pixels are unlabeled are waiting for prediction (Step 2). The next step will divide the remaining 
    windows with more than one category (including the unclassified ones) into training and test data according to a predefined order. 
    The predefined order can be either by category or by the number of samples within each category." ... "the predefined order by 
    category as an example. [] multiclass windows containing the first class are collected; then, a predefined proportion of windows is 
    randomly selected for training, while the remaining windows are used for testing. Afterward, the labels of all pixels in windows 
    containing the first class are set to zero to avoid repeated sampling. The remaining windows are then used to collect the second 
    category. This process is repeated until sampling is complete for all classes"

    Observations/Notes:
    - The predefined order sorting by category is not exactly clear, specifically what the ordering of classes is to select multiclass 
    windows by. In this implementation we sort lowest to highest count of class representation within multiclass windows.
    - It is assumed that statement "the labels of all pixels in windows containing the first class are set to zero to avoid repeated 
    sampling." refers to how the algorithm is implemented and not a process of the algorithm itself.
    - Two alterations are mentioned that are not implemented here. The first is to flip the train and test sets, that is, train on
    uni-class windows. The second is to simply split multiclass windows random between the train and test sets. Zhange noted that the
    first alteration provides poor model performance.
    """
    def __init__(self,
                 image: np.ndarray, labels: np.ndarray, classes: Dict[int, str],
                 patch_size: Tuple[int, int], train_ratio: float, 
                 name: str, random_seed: Optional[int] = None, **kwargs) -> None:
        super().__init__(image, labels, classes, patch_size, train_ratio, name, random_seed, **kwargs)

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

        # Sort the multiclass window class counts lowest to highest
        indices = np.arange(len(self.classes))
        class_index_order = indices[np.argsort(multiclass_class_counts)]

        # Split multiclass windows into train and test
        to_process_windows = multiclass_windows
        training_cpls_per_class = {k: [] for k in self.classes.keys()}
        for class_idx in class_index_order:
            if len(to_process_windows) == 0: break

            # Get all multiclass windows with this class present in it
            windows = []
            unused_windows = []
            for y, x, cpl_class, class_occupancy in to_process_windows:
                if class_occupancy[class_idx] == 1:
                    windows.append((y, x, cpl_class, class_occupancy))
                else:
                    unused_windows.append((y, x, cpl_class, class_occupancy))
            
            # Update the windows to process to the ones we didn't select
            to_process_windows = unused_windows

            # Split windows based on train_ratio
            total_count = len(windows)
            if total_count > 0:
                train_count = int(self.train_ratio * total_count)
                train_inds = np.random.choice(total_count, size=train_count, replace=False)
                test_inds = np.setdiff1d(np.arange(total_count), train_inds)

                # Insert into training cpls
                for idx in train_inds:
                    y, x, cpl_class, _ = windows[idx]
                    training_cpls_per_class[cpl_class].append((y, x))

                # Insert into testing cpls
                for idx in test_inds:
                    y, x, cpl_class, _ = windows[idx]
                    testing_cpls_per_class[cpl_class].append((y, x))
        
        # Ensure we worked through all multiclass windows
        if len(to_process_windows) > 0:
            raise RuntimeError(f'! Did not sort all multiclass windows !')
        
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
