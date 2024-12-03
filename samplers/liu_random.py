import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

from samplers._overlap import cpl_validity_numba
from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerLiuRandom(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: False

    Article Bibtex Reference:
        @article{Liu2022b,
            abstract = {Convolutional neural networks (CNNs) are widely used in hyperspectral image (HSI) classification. However, 
            the network architecture of CNNs is often designed manually, which requires careful fine-tuning. Recently, many techniques 
            for neural architecture search (NAS) have been proposed to design the network automatically but most of the methods 
            are only concerned with the overall classification accuracy and ignore the balance between the floating point operations
              per second (FLOPs) and the number of parameters. In this article, we propose a new multiobjective optimization (MO) 
              method called MO-CNN to automatically design the structure of CNNs for HSI classification. First, a MO method based 
              on continuous particle swarm optimization (CPSO) is constructed, where the overall accuracy, FLOPs, and the number 
              of parameters are considered, to obtain an optimal architecture from the Pareto front. Then, an auxiliary skip 
              connection strategy is added (together with a partial connection strategy) to avoid performance collapse and to 
              reduce memory consumption. Furthermore, an end-to-end band selection network (BS-Net) is used to reduce redundant 
              bands and to maintain spectral-spatial uniformity. To demonstrate the performance of our newly proposed MO-CNN in 
              scenarios with limited training sets, a quantitative and comparative analysis (including ablation studies) is conducted.
                Our optimization strategy is shown to improve the classification accuracy, reduce memory, and obtain an optimal 
                structure for CNNs based on unbiased datasets.},
            author = {Xiaobo Liu and Xin Gong and Antonio Plaza and Zhihua Cai and Xiao Xiao and Xinwei Jiang and Xiang Liu},
            doi = {10.1109/TGRS.2022.3220748},
            issn = {15580644},
            journal = {IEEE Transactions on Geoscience and Remote Sensing},
            keywords = {Convolutional neural networks (CNNs),SuperNet,hyperspectral images (HSIs),
            multiobjective optimization (MO),neural architecture search (NAS)},
            publisher = {Institute of Electrical and Electronics Engineers Inc.},
            title = {MO-CNN: Multiobjective Optimization of Convolutional Neural Networks for Hyperspectral Image Classification},
            volume = {60},
            year = {2022},
        }
    
    Article Sampler Description Excerpt:
    Section IV.A "randomly choose one pixel belonging to one class as the central pixel and obtain training samples by region 
    extension. We choose 8x8 images by a region extended from one pixel to 8x8 images for unbiased datasets and obtain 8x8 images 
    by a region extended for the training set. The 8x8 image-by-region extension means finding one pixel as the central point and 
    choosing the pixels around the point to form the training sample. For the testing data, we choose the data that do not belong 
    to the training dataset. First, we set the index of all the pixels to 1, and the index is set to 0 once the pixel has been 
    selected. Before we sample the center pixel, we guarantee that the index around the center is 1. If the index around the center 
    pixel is 0, we will modify the center pixel. In this way, we can guarantee that none of the samples are overlapping."

    Observations/Notes:
    - It is unclear what is meant by the phrases "region extension", "images", and "image-by-region". It is assumed that this collectively
    is a misunderstanding or interpretation of the referenced Liang Controlled Random Sampling. Specifically Liangs sub-algorithm of 
    region growing from a seed pixel is misunderstood to mean the practice of slicing an image to obtain a patch/sample. Further, the
    usage and connection of the words "images", "image-by-region", and "samples" moderately supports this observation.
    - Thusly, this implementation is simply random sampling with a minimum distance enforced between train and test samples to
    ensure no overlap is preesent.
    - It is never stated that multiple training samples are selected, only that "8x8 images" are selected for each class. Though, in the 
    same paragraph it is noted that 200 samples are used. It is assumed then that this is Random Equal Count sampling without overlap.
    """
    def __init__(self,
                 image: np.ndarray, labels: np.ndarray, classes: Dict[int, str],
                 patch_size: Tuple[int, int], train_ratio: float, 
                 name: str, random_seed: Optional[int] = None, **kwargs) -> None:
        super().__init__(image, labels, classes, patch_size, train_ratio, name, random_seed, **kwargs)

        # Default training_class_sample_count to 100 if not specified
        self.training_class_sample_count = kwargs.get('training_class_sample_count', 50)

    @numpy_deterministic
    def sample(self) -> None:
        print(f'> Executing: {self.name} ({self.__class__.__name__})')

        # Get the initial set of valid center pixel locations for each class
        valid_cpls_per_class, _ = self._initial_cpls_per_class()

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

        # Calculate the valid cpls in relation to the selected training cpls
        flat_train_cpls = np.concatenate(list(training_cpls_per_class.values()), axis=0)
        flat_test_cpls = np.concatenate(list(testing_cpls_per_class.values()), axis=0)
        _, _, _, _, _, valid_test_cpls, _ = cpl_validity_numba(self.image.shape, flat_train_cpls, flat_test_cpls, self.patch_bboxes)

        # Set and Sort the cpls back to per class
        test_cpls = valid_test_cpls
        test_cpls_per_class = self._sort_cpls(test_cpls)

        # Set the cpls
        self.training_cpls = training_cpls_per_class
        self.testing_cpls = test_cpls_per_class
