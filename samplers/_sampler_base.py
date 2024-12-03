"""
The base class for all samplers. 

To get it out of the way now. This is not a torch.utils.data.Sampler type of class.
Pytorch's Sampler is about selecting pre-made samples. This class is about producing
samples. Same name, different action.

Subclasses of this class should be standalone from the Pytorch codebase for users who
do not wish to use Pytorch. They should be able to just ingest data and produces samples.
"""
import os
import sys
import json
import functools
import numpy as np
from collections import deque

from abc import ABC, abstractmethod
from typing import Generator, List, Optional, Dict, Tuple, Any, TypeVar, Union


def numpy_deterministic(func):
    """ A decorator to control numpy RNG state within BaseImageSampler.sample methods. """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Try to get calling instance random_seed if it exists
        seed = self.random_seed if hasattr(self, 'random_seed') else None
        if seed:
            # Save the current state of numpy's RNG
            state = np.random.get_state()
            # Set the seed from the instance's attribute
            np.random.seed(seed)
            try:
                # Call the wrapper function
                result = func(self, *args, **kwargs)
            finally:
                # Reset the RNG to its starting state
                np.random.set_state(state)
            return result
        else:
            # If no random_seed provided, do nothing.
            return func(self, *args, **kwargs)
    return wrapper


BaseImageSamplerType = TypeVar('BaseImageSamplerType', bound='BaseImageSampler')

class BaseImageSampler(ABC):

    def __init__(self,
                 image: np.ndarray, labels: np.ndarray, classes: Dict[int, str],
                 patch_size: Tuple[int, int], train_ratio: float,
                 name: str, random_seed: Optional[int] = None, **kwargs) -> None:
        # Common attributes to all samplers
        self.image: np.ndarray = image
        self.labels: np.ndarray = labels
        self.classes: Dict[int, str] = classes
        self.patch_size: Tuple[int, int] = patch_size
        self.train_ratio: float = train_ratio

        # Useful attributes, kwargs for sampler specific arguments
        self.name: str = name
        self.random_seed: Optional[int] = random_seed
        self.kwargs: Dict[str, Any] = kwargs

        # The (y, x) location of the center pixel locations of samples per class. (May not contain all classes!)
        self.training_cpls: Dict[int, np.ndarray] = None   # {class: (num_cpls, (y, x)) }
        self.testing_cpls: Dict[int, np.ndarray] = None    # {class: (num_cpls, (y, x)) }

        # If generated, the cpls for the training_cpls split into training and validation
        self.split_training_cpls: Dict[int, np.ndarray] = None   # {class: (num_cpls, (y, x)) }
        self.split_validation_cpls: Dict[int, np.ndarray] = None   # {class: (num_cpls, (y, x)) }

        # Helper data structures. The "bboxes" of sample and overlap slices w.r.t each pixel location.
        self.valid_cpl_range: np.ndarray = np.array([0, 0, self.labels.shape[0], self.labels.shape[1]])
        self.patch_bboxes: np.ndarray = np.zeros((self.labels.shape[0], self.labels.shape[1], 4), dtype=np.int32)
        self.over_bboxes: np.ndarray = np.zeros((self.labels.shape[0], self.labels.shape[1], 4), dtype=np.int32)
        self._calculate_helper_structures()
    
    def _calculate_helper_structures(self, clip_to_edge: bool = True):
        """
        Calculate the "bounding boxes" of all possible patches and overlap boundary for every pixel location.

        This function handles all positive valued patch sizes. This includes all square, rectangular, odd, and 
        even valued components of the patch size. For example, 1x1, 3x3, 4x4, 5x2, 4x7 are all valid patch sizes.
        It does this by making the decision that when a center pixel does not fall on a discrete pixel location
        (even valued dimension) then the cpl is selected as the bottom right possible location of the 4 total
        possible selections. 

        This function also handles edges of images by either clipping bboxes to the edges, or calculating
        indices that may be negative or greater than pixel space. This is done so that users of the helper
        structures can calculate padding for patches created at the edge of images.
        """
        # The size of the pixel space
        height, width = self.labels.shape

        # The distance from the CPL we go in either direction to get to the patch edge
        patch_half_y, patch_half_x = self.patch_size[0] // 2, self.patch_size[1] // 2

        # An adjustment for odd valued patch dimension to get to the edge of the patch
        patch_odd_adjust_y, patch_odd_adjust_x = self.patch_size[0] % 2, self.patch_size[1] % 2

        # An adjustment for even valued patch dimensions to get to the edge of the overlap boundary
        overlap_even_adjust_y, overlap_even_adjust_x = 1 - self.patch_size[0] % 2, 1 - self.patch_size[1] % 2

        # Calculate the valid range for cpls that do not result in sub_patch sized patches
        cpl_min_y, cpl_min_x = patch_half_y, patch_half_x
        cpl_max_y = height - (patch_half_y + patch_odd_adjust_y)
        cpl_max_x = width - (patch_half_x + patch_odd_adjust_x)
        self.valid_cpl_range = [cpl_min_y, cpl_min_x, cpl_max_y, cpl_max_x]

        # For all pixel locations calculate the patch bbox coordinates and the overlap boundary coordinates
        for y in range(height):
            for x in range(width):
                # Calculate the top left and bottom right coordinates of the patch
                ty = y - patch_half_y
                tx = x - patch_half_x 
                by = ty + patch_half_y * 2 + patch_odd_adjust_y
                bx = tx + patch_half_x * 2 + patch_odd_adjust_x
                # Do not overrun the edge of the pixel space
                if clip_to_edge:
                    ty, tx = max(0, ty), max(0, tx)
                    by, bx = min(height, by), min(width, bx)
                self.patch_bboxes[y, x, ...] = np.array([ty, tx, by, bx])

                # Calculate the top left and bottom right coordinates of the overlap boundary
                ty = y - patch_half_y * 2 + overlap_even_adjust_y 
                tx = x - patch_half_x * 2 + overlap_even_adjust_x 
                by = ty + patch_half_y * 4 + patch_odd_adjust_y - overlap_even_adjust_y 
                bx = tx + patch_half_x * 4 + patch_odd_adjust_x - overlap_even_adjust_x
                # Do not overrun the edge of the pixel space
                if clip_to_edge:
                    ty, tx = max(0, ty), max(0, tx)
                    by, bx = min(height, by), min(width, bx)
                self.over_bboxes[y, x, ...] = np.array([ty, tx, by, bx])
    
    @abstractmethod
    def sample(self) -> None:
        pass

    @numpy_deterministic
    def split_training_to_validation(self, ratio_of_train: float = 0.15):
        # Flatten the training cpls into a single array and shuffle them
        flat_cpls = np.concatenate(list(self.training_cpls.values()), axis=0)
        np.random.shuffle(flat_cpls)

        # Calculate the number of samples to place into validation
        num_samples = len(flat_cpls)
        val_samples = int(num_samples * ratio_of_train)
        val_samples = max(1, val_samples)

        if num_samples < 2:
            raise RuntimeError(f'! Cannot split training cpls, too small !')
        
        # Split the training cpls into validation and test
        val_cpls = flat_cpls[:val_samples]
        train_cpls = flat_cpls[val_samples:]

        # Sort cpls back to per class structure.
        self.split_training_cpls = self._sort_cpls(train_cpls)
        self.split_validation_cpls = self._sort_cpls(val_cpls)

    def _slice_from_center_pixels(self, cpls: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Given a set of center pixel locations by class slice the image and labels """
        # Flatten to a single list and ensure data type
        cpls = cpls.astype(np.int32)

        # Preallocate an array for samples
        num_cpls = len(cpls)
        image_slices = np.zeros((num_cpls, *self.patch_size, self.image.shape[-1]))
        labels_slices = np.zeros((num_cpls, *self.patch_size))
        # slice_locations = np.zeros((num_cpls, 2))

        # Slice the image and labels 
        for slice_idx, (y, x) in enumerate(cpls):
            ty, tx, by, bx = self.patch_bboxes[y, x]
            image_slices[slice_idx] = self.image[ty:by, tx:bx]
            labels_slices[slice_idx] = self.labels[ty:by, tx:bx]
            # slice_locations[slice_idx] = [top_left_x, top_left_y]
        
        return image_slices, labels_slices

    def _initial_cpls_per_class(self) -> Tuple[Dict[int, np.ndarray], int]:
        """
        Get the set of all initially valid center pixel locations on which samples can be centered.
        Validity of CPLs is determined by the precomputed valid CPL range ensuring they are at least 
        one-half patch size from the edge and not belonging to the unlabelled class.
        """
        # Retrieve valid CPL ranges precomputed in _calculate_helper_structures
        cpl_min_y, cpl_min_x, cpl_max_y, cpl_max_x = self.valid_cpl_range

        # Define the ranges of y and x within which to consider the center pixels
        valid_rows = range(cpl_min_y, cpl_max_y + 1)
        valid_cols = range(cpl_min_x, cpl_max_x + 1)

        # Filter and categorize valid CPLs by class, excluding the 'unlabelled' class
        valid_cpls_per_class = {}
        for class_idx in self.classes.keys():
            if self.classes[class_idx] == 'unlabelled': continue

            # Create a mask where the class label matches and within valid rows and columns
            class_mask = (self.labels[valid_rows, :][:, valid_cols] == class_idx)

            # Find valid indices where class_mask is True (Adjust indices to original image coordinates)
            valid_y, valid_x = np.where(class_mask)
            valid_y += cpl_min_y
            valid_x += cpl_min_x

            # Store these adjusted coordinates
            valid_cpls_per_class[class_idx] = np.column_stack((valid_y, valid_x))

        total_count = sum(len(cpls) for cpls in valid_cpls_per_class.values())
        return valid_cpls_per_class, total_count

    def _filter_cpls_by_image_edge_patch_size(self, cpls: np.ndarray) -> np.ndarray:
        """ Filter center pixel locations based on their distance from the edge of the image. """
        # Retrieve precomputed valid ranges for CPLs
        cpl_min_y, cpl_min_x, cpl_max_y, cpl_max_x = self.valid_cpl_range

        # Get the y and x coordinates of the center pixel locations
        y_coords, x_coords = cpls[:, 0], cpls[:, 1]

        # Check if the center pixel locations are within the valid range
        valid_mask = (y_coords >= cpl_min_y) & (y_coords <= cpl_max_y) & \
                    (x_coords >= cpl_min_x) & (x_coords <= cpl_max_x)

        # Filter out center pixel locations that are too close to the edge
        filtered_cpls = cpls[valid_mask]

        return filtered_cpls
    
    def _filter_cpls_by_split_line(self, cpls: np.ndarray, split_line: Tuple[float, float]) -> np.ndarray:
        """ Filter center pixel locations based on their distance from a split line. """
        # TODO: Need to check if this makes sense for rectangular and even valued patch sizes.

        # Get the y and x coordinates of the center pixel locations
        y_coords, x_coords = cpls[:, 0], cpls[:, 1]

        # Calculate the distance of each center pixel location from the split line
        distances = np.abs(y_coords - split_line[0] * x_coords - split_line[1]) / np.sqrt(1 + split_line[0] ** 2)

        # Get the maximum distance allowed from the split line based on the patch size
        max_distance = np.sqrt((self.patch_size[0] / 2) ** 2 + (self.patch_size[1] / 2) ** 2)

        # Check if the center pixel locations are at least half the patch size away from the split line
        valid_mask = distances >= max_distance

        # Filter out center pixel locations that are too close to the split line
        filtered_cpls = cpls[valid_mask]

        return filtered_cpls
    
    # @numpy_deterministic # Assume calling method (self.sample) is decorated with numpy_deterministic.
    def _randomsplit_sort_set_cpls(self, cpls: np.ndarray) -> None:
        """ 
        Given an array of center pixel locations representing all samples, split into train and test 
        and also sort by class into a Dict[int, cpl.ndarray] attribute.
        """
        total_count = cpls.shape[0]
        train_count = int(self.train_ratio * total_count)
        train_inds = np.random.choice(total_count, size=train_count, replace=False)
        test_inds = np.setdiff1d(np.arange(total_count), train_inds)

        train_cpls = cpls[train_inds]
        test_cpls = cpls[test_inds]

        self.training_cpls = self._sort_cpls(train_cpls)
        self.testing_cpls = self._sort_cpls(test_cpls)

    def _sort_cpls(self, cpls: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Sort a list of center pixel locations into their respective classes. 
        """
        cpls = cpls.astype(np.int32)  # Ensure int32
        cpls_per_class = {k: np.empty((0, 2), dtype=int) for k in self.classes.keys()}
        all_center_y, all_center_x = cpls.T
        for class_idx in self.classes.keys():
            class_mask = self.labels[all_center_y, all_center_x] == class_idx
            class_cpls = cpls[class_mask]
            cpls_per_class[class_idx] = class_cpls
        return cpls_per_class

    def save(self, filename: str) -> None:
        # Create a dictionary to store the attributes
        data: Dict[str, Any] = {
            'name': self.name,
            'type': str(type(self)),
            'patch_size': self.patch_size,
            'train_ratio': self.train_ratio,
            'random_seed': self.random_seed,
            'kwargs': list(self.kwargs.keys()),  # Save only the keys of kwargs
            'image_shape': self.image.shape,
            'labels_shape': self.labels.shape,
            'classes': self.classes
        }

        # Save training_cpls and testing_cpls
        if self.training_cpls is not None:
            data['training_cpls'] = {class_idx: cpls.tolist() for class_idx, cpls in self.training_cpls.items()}
        if self.testing_cpls is not None:
            data['testing_cpls'] = {class_idx: cpls.tolist() for class_idx, cpls in self.testing_cpls.items()}

        # Save training_cpls and validation_cpls if they were split
        if self.split_training_cpls is not None:
            data['split_training_cpls'] = {class_idx: cpls.tolist() for class_idx, cpls in self.split_training_cpls.items()}
        if self.split_validation_cpls is not None:
            data['split_validation_cpls'] = {class_idx: cpls.tolist() for class_idx, cpls in self.split_validation_cpls.items()}

        # Write data to JSON file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self, filename: str) -> None:
        # Load data from JSON file
        with open(filename, 'r') as f:
            data = json.load(f)

        # Restore attributes
        self.name = data['name']
        self.patch_size = tuple(data['patch_size'])
        self.train_ratio = data['train_ratio']
        self.random_seed = data['random_seed']
        self.kwargs = {key: None for key in data['kwargs']}  # Populate kwargs with None values
        
        # Compare attributes to class instance
        class_type_str = data['type']
        kwargs = data['kwargs']
        image_shape = data['image_shape']
        labels_shape = data['labels_shape']
        classes = data['classes']

        if str(type(self)) != class_type_str:
            raise RuntimeError(f'! Loaded class type string does not match instance type(self) !')
        if sum([1 for k in self.kwargs if str(k) not in kwargs]) > 1:
            pass
        if self.image.shape != image_shape:
            raise RuntimeError(f'! Loaded image shape does not match instance image.shape !')
        if self.labels.shape != labels_shape:
            raise RuntimeError(f'! Loaded image shape does not match instance image.shape !')
        if classes != self.classes:
            raise RuntimeError(f'! Loaded classes do not match instance classes !')

        # Restore training_cpls and testing_cpls
        if 'training_cpls' in data:
            self.training_cpls = {int(class_idx): np.array(cpls) for class_idx, cpls in data['training_cpls'].items()}
        if 'testing_cpls' in data:
            self.testing_cpls = {int(class_idx): np.array(cpls) for class_idx, cpls in data['testing_cpls'].items()}
        
        # Restore training_cpls and validation_cpls if they were split
        if 'split_training_cpls' in data:
            self.split_training_cpls = {int(class_idx): np.array(cpls) for class_idx, cpls in data['split_training_cpls'].items()}
        if 'testing_cpls' in data:
            self.split_validation_cpls = {int(class_idx): np.array(cpls) for class_idx, cpls in data['split_validation_cpls'].items()}
    
    @staticmethod
    def find_connected_components(pixel_locations: np.ndarray) -> List[np.ndarray]:
        """ Identify all connected components within the given pixel locations. """
        if not len(pixel_locations):
            return []

        # Prepare to collect components
        components = []
        remaining_pixels = set(map(tuple, pixel_locations))
        while remaining_pixels:
            seed_pixel = remaining_pixels.pop()  # Get an arbitrary seed pixel
            seed_index = np.where((pixel_locations == seed_pixel).all(axis=1))[0][0]

            # Grow region from the seed to include all possible connected pixels
            region, leftovers, _ = BaseImageSampler.grow_region(pixel_locations, seed_pixel_ind=seed_index, region_ratio=1.0)
            
            # Append the found component
            components.append(region)
            
            # Update remaining pixels
            remaining_pixels.difference_update(map(tuple, region))
            #remaining_pixels = set(map(tuple, leftovers))

        return components
    
    @staticmethod
    def grow_region(pixel_locations: np.ndarray, seed_pixel_ind: int = None, region_ratio: Union[float, None] = None, target_region_size: Union[float, None] = None) -> Tuple[np.ndarray, np.ndarray, bool]:
        """ Grow a connected region from pixel_locations up to a specified ratio of its total size. """
        if pixel_locations.size == 0:
            return np.array([]), np.array([]), True  # Handle empty input
        
        # Initialize set of pixel tuples for fast lookup
        pixel_tuples = set(map(tuple, pixel_locations))

        # # Determine the size of the region to grow
        # target_region_size = int(region_ratio * len(pixel_tuples))
        # if target_region_size == 0:
        #     return np.array([]), pixel_locations.copy(), True
        
        # Determine the target size (count) of the region
        if region_ratio is not None:
            # Based on train_ratio
            target_region_size = int(region_ratio * len(pixel_tuples))
        elif target_region_size is not None:
            # Based on a count (probably stratified)
            pass
        else:
            raise ValueError(f'Must provide either region_ratio or target_region_size!')
            # Default to using the entire region if neither given (could just retu)
            # target_region_size = len(pixel_tuples)

        # Do nothing if there's nothing to do
        if target_region_size == 0:
            return np.array([]), pixel_locations.copy(), True

        # Select seed pixel
        if seed_pixel_ind is None:
            # Assume calling method (self.sample) is decorated with numpy_deterministic.
            seed_pixel_ind = np.random.randint(len(pixel_locations))
        seed = tuple(pixel_locations[seed_pixel_ind])
        
        # Grow the region using a breadth-first search
        region_set = set([seed])
        queue = deque([seed])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Define directions once to avoid recreating them
        while queue and len(region_set) < target_region_size:
            y, x = queue.popleft()
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                neighbor = (ny, nx)
                if neighbor in pixel_tuples and neighbor not in region_set:
                    region_set.add(neighbor)
                    queue.append(neighbor)
        
        # Prepare output arrays
        region_array = np.array(list(region_set), dtype=int)
        remaining_set = pixel_tuples - region_set
        remaining_array = np.array(list(remaining_set), dtype=int)
        
        # Check if the region reached the desired size
        reached_target_size = len(region_set) == target_region_size

        return region_array, remaining_array, reached_target_size
