import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

from samplers._overlap import cpl_validity_numba
from samplers._sampler_base import BaseImageSampler, numpy_deterministic


class SamplerAcquarelliControlled(BaseImageSampler):
    """
    (This Implementation) Allows Overlap: False

    Article Bibtex Reference:
        @article{Acquarelli2018,
            abstract = {Spectral-spatial classification of hyperspectral images has been the subject of many studies in recent
            years. When there are only a few labeled pixels for training and a skewed class label distribution, this task 
            becomes very challenging because of the increased risk of overfitting when training a classifier. In this paper,
            we show that in this setting, a convolutional neural network with a single hidden layer can achieve 
            state-of-the-art performance when three tricks are used: a spectral-locality-aware regularization term and 
            smoothing- and label-based data augmentation. The shallow network architecture prevents overfitting in the 
            presence of many features and few training samples. The locality-aware regularization forces neighboring 
            wavelengths to have similar contributions to the features generated during training. The new data 
            augmentation procedure favors the selection of pixels in smaller classes, which is beneficial for skewed 
            class label distributions. The accuracy of the proposed method is assessed on five publicly available 
            hyperspectral images, where it achieves state-of-the-art results. As other spectral-spatial classification 
            methods, we use the entire image (labeled and unlabeled pixels) to infer the class of its unlabeled pixels. 
            To investigate the positive bias induced by the use of the entire image, we propose a new learning setting 
            where unlabeled pixels are not used for building the classifier. Results show the beneficial effect of the 
            proposed tricks also in this setting and substantiate the advantages of using labeled and unlabeled pixels 
            from the image for hyperspectral image classification.},
            author = {Jacopo Acquarelli and Elena Marchiori and Lutgarde M.C. Buydens and Thanh Tran and Twan van Laarhoven},
            doi = {10.3390/RS10071156},
            issn = {2072-4292},
            issue = {7},
            journal = {Remote Sensing 2018, Vol. 10, Page 1156},
            keywords = {convolutional neural networks,data augmentation,hyperspectral images,learning setting},
            month = {7},
            pages = {1156},
            publisher = {Multidisciplinary Digital Publishing Institute},
            title = {Spectral-Spatial Classification of Hyperspectral Images: Three Tricks and a New Learning Setting},
            volume = {10},
            url = {https://www.mdpi.com/2072-4292/10/7/1156/htm https://www.mdpi.com/2072-4292/10/7/1156},
            year = {2018},
        }
    
    Article Sampler Description Excerpt:
    Section 2.2.2 "In particular, we propose to randomly select a single patch of pixels for each class to use as training data. 
    We use a patch of 7x7 labeled pixels for each class as a training set, which ensures that we have enough 
    training pixels (at most 49) per class."

    Observations/Notes:
    - While it is not directly stated, it is assumed based on Acquarelli's understanding of sampling that this method does 
    not allow overlap and thus it is enforced by removing any offending test CPLs from the final test set.
    - This method will be limited in the amount of training data it can produce overall. While this implementation
    allows for any patch size, as described the 7x7 patch size will result in extremely small training data sets especially
    for larger data sets.
    - This method will also always be limited by the smallest dimension of the class partition(s). If a class does not 
    contain a patch_size chunk entirely of class_idx, then we cannot generate training data for that class.
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

        # Initially valid cpls for both train and test
        init_test_cpls_per_class = valid_cpls_per_class
        init_train_cpls_per_class = valid_cpls_per_class

        # Randomly select the first train cpl that is entirely composed of cpl's class_idx
        train_cpls_per_class = {}
        for class_idx, class_cpls in init_train_cpls_per_class.items():
            # Shuffle class cpls
            np.random.shuffle(class_cpls)
            
            # Variables to track the patch with the most class_idx labels
            best_cpl = None
            max_class_idx_count = -1
            for y, x in class_cpls:
                ty, tx, by, bx = self.patch_bboxes[y, x]
                label_slice = self.labels[ty:by, tx:bx]
                class_idx_count = np.sum(label_slice == class_idx)
                
                # If the patch is entirely of class_idx, use it and stop searching
                if class_idx_count == label_slice.size:
                    train_cpls_per_class[class_idx] = np.array([[y, x]])
                    break
                
                # Otherwise, track the patch with the most class_idx labels
                if class_idx_count > max_class_idx_count:
                    max_class_idx_count = class_idx_count
                    best_cpl = np.array([[y, x]])
            
            # If no patch was entirely class_idx, use the best found patch
            if class_idx not in train_cpls_per_class and best_cpl is not None:
                train_cpls_per_class[class_idx] = best_cpl
                print(f'! WARNING: Could not locate fully matching patch for class: {class_idx} - {self.classes[class_idx]}. Using patch with most matching labels.')

        # Calculate the valid cpls in relation to the selected training cpls
        flat_train_cpls = np.concatenate(list(train_cpls_per_class.values()), axis=0)
        flat_test_cpls = np.concatenate(list(init_test_cpls_per_class.values()), axis=0)
        _, _, _, _, _, valid_test_cpls, _ = cpl_validity_numba(self.image.shape, flat_train_cpls, flat_test_cpls, self.patch_bboxes)

        # Set and Sort the cpls back to per class
        test_cpls = valid_test_cpls
        test_cpls_per_class = self._sort_cpls(test_cpls)

        # Set the cpls
        self.training_cpls = train_cpls_per_class
        self.testing_cpls = test_cpls_per_class
