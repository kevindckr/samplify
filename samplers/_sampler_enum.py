from enum import Enum, auto
from typing import Dict, List, Tuple, Type, TypeVar

from datasets._dataset_enum import DatasetType
from datasets._dataset_base import BaseImageDataset

from samplers._sampler_base import BaseImageSampler


from samplers.acquarelli_controlled import SamplerAcquarelliControlled
from samplers.clark_stratified import SamplerClarkStratified
from samplers.decker_dynamic import SamplerDeckerDynamic
from samplers.decker_spatial import SamplerDeckerSpatial
from samplers.grid import SamplerGrid
from samplers.grss13_suggested import SamplerGRSS13Suggested
from samplers.hansch_cluster import SamplerHanschCluster
from samplers.lange_controlled import SamplerLangeControlled
from samplers.liang_controlled import SamplerLiangControlled
from samplers.liu_random import SamplerLiuRandom
from samplers.random_equal_count import SamplerRandomEqualCount
from samplers.random_stratified import SamplerRandomStratified
from samplers.random_uniform import SamplerRandomUniform
from samplers.simple_spatial_partitioning import SamplerSimpleSpatialPartitioning
from samplers.zhang_grid import SamplerZhangGrid
from samplers.zhou_controlled import SamplerZhouControlled
from samplers.zou_grid import SamplerZouGrid


SamplerTypeType = TypeVar('SamplerTypeType', bound='SamplerType')

class SamplerType(Enum):
    """ A helper Enum to wrap BaseImageSampler subclasses. """
    ACQUARELLI_CONTROLLED = auto()
    CLARK_STRATIFIED = auto()
    DECKER_DYNAMIC = auto()
    DECKER_SPATIAL = auto()
    GRID = auto()
    GRSS13_SUGGESTED = auto()
    HANSCH_CLUSTER = auto()
    LANGE_CONTROLLED = auto()
    LIANG_CONTROLLED = auto()
    LIU_RANDOM = auto()
    RANDOM_EQUAL_COUNT = auto()
    RANDOM_STRATIFIED = auto()
    RANDOM_UNIFORM = auto()
    SPATIAL_SIMPLE = auto()
    ZHANG_GRID = auto()
    ZHOU_CONTROLLED = auto()
    ZOU_GRID = auto()

    @staticmethod
    def instantiate_samplers(datasets: List[BaseImageDataset], patch_size: Tuple[int, int] = (32, 32), train_ratio: float = 0.7
                             ) -> Dict[DatasetType, Dict[SamplerTypeType, BaseImageSampler]]:
        """ Instantiate every sampler type for each provided dataset. """
        dataset_samplers = {}
        for dataset in datasets:
            kwargs = {'dataset': dataset}
            dataset_type = DatasetType.get_dataset_type(dataset)

            # Add the special kwargs for SamplerGRSS13Suggested (the other samplers will ignore this)
            if dataset_type == DatasetType.GRSS13:
                kwargs = {'dataset': dataset, 'labels_train': dataset.labels_train, 'labels_test': dataset.labels_test}

            dataset_samplers[dataset_type] = {}
            for sampler_type in SamplerType:
                # Do not instantiate a GRSS13_SUGGESTED sampler for a non-GRSS13 dataset.
                if dataset_type != DatasetType.GRSS13 and sampler_type == SamplerType.GRSS13_SUGGESTED:
                    continue
                # Get the sampler class and instantiate.
                sampler_class = SamplerType.get_sampler_class(sampler_type)
                sampler_instance = sampler_class(
                    dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio,
                    sampler_type.name, None, **kwargs)
                dataset_samplers[dataset_type][sampler_type] = sampler_instance
        return dataset_samplers

    @staticmethod
    def get_sampler_class(sampler_type: SamplerTypeType) -> Type[BaseImageSampler]:
        """ Enum to Class """
        if sampler_type == SamplerType.ACQUARELLI_CONTROLLED:
            return SamplerAcquarelliControlled
        elif sampler_type == SamplerType.CLARK_STRATIFIED:
            return SamplerClarkStratified
        elif sampler_type == SamplerType.DECKER_DYNAMIC:
            return SamplerDeckerDynamic
        elif sampler_type == SamplerType.DECKER_SPATIAL:
            return SamplerDeckerSpatial
        elif sampler_type == SamplerType.GRID:
            return SamplerGrid
        elif sampler_type == SamplerType.GRSS13_SUGGESTED:
            return SamplerGRSS13Suggested
        elif sampler_type == SamplerType.HANSCH_CLUSTER:
            return SamplerHanschCluster
        elif sampler_type == SamplerType.LANGE_CONTROLLED:
            return SamplerLangeControlled
        elif sampler_type == SamplerType.LIANG_CONTROLLED:
            return SamplerLiangControlled
        elif sampler_type == SamplerType.LIU_RANDOM:
            return SamplerLiuRandom
        elif sampler_type == SamplerType.RANDOM_EQUAL_COUNT:
            return SamplerRandomEqualCount
        elif sampler_type == SamplerType.RANDOM_STRATIFIED:
            return SamplerRandomStratified
        elif sampler_type == SamplerType.RANDOM_UNIFORM:
            return SamplerRandomUniform
        elif sampler_type == SamplerType.SPATIAL_SIMPLE:
            return SamplerSimpleSpatialPartitioning
        elif sampler_type == SamplerType.ZHANG_GRID:
            return SamplerZhangGrid
        elif sampler_type == SamplerType.ZHOU_CONTROLLED:
            return SamplerZhouControlled
        elif sampler_type == SamplerType.ZOU_GRID:
            return SamplerZouGrid
        else:
            raise ValueError(f'Unknown sampler type: {sampler_type.name}')
    
    @staticmethod
    def get_sampler_type(sampler_instance: BaseImageSampler) -> SamplerTypeType:
        """ Class to Enum """
        if isinstance(sampler_instance, SamplerAcquarelliControlled):
            return SamplerType.ACQUARELLI_CONTROLLED
        elif isinstance(sampler_instance, SamplerClarkStratified):
            return SamplerType.CLARK_STRATIFIED
        elif isinstance(sampler_instance, SamplerDeckerDynamic):
            return SamplerType.DECKER_DYNAMIC
        elif isinstance(sampler_instance, SamplerDeckerSpatial):
            return SamplerType.DECKER_SPATIAL
        elif isinstance(sampler_instance, SamplerGrid):
            return SamplerType.GRID
        elif isinstance(sampler_instance, SamplerGRSS13Suggested):
            return SamplerType.GRSS13_SUGGESTED
        elif isinstance(sampler_instance, SamplerHanschCluster):
            return SamplerType.HANSCH_CLUSTER
        elif isinstance(sampler_instance, SamplerLangeControlled):
            return SamplerType.LANGE_CONTROLLED
        elif isinstance(sampler_instance, SamplerLiangControlled):
            return SamplerType.LIANG_CONTROLLED
        elif isinstance(sampler_instance, SamplerLiuRandom):
            return SamplerType.LIU_RANDOM
        elif isinstance(sampler_instance, SamplerRandomEqualCount):
            return SamplerType.RANDOM_EQUAL_COUNT
        elif isinstance(sampler_instance, SamplerRandomStratified):
            return SamplerType.RANDOM_STRATIFIED
        elif isinstance(sampler_instance, SamplerRandomUniform):
            return SamplerType.RANDOM_UNIFORM
        elif isinstance(sampler_instance, SamplerSimpleSpatialPartitioning):
            return SamplerType.SPATIAL_SIMPLE
        elif isinstance(sampler_instance, SamplerZhangGrid):
            return SamplerType.ZHANG_GRID
        elif isinstance(sampler_instance, SamplerZhouControlled):
            return SamplerType.ZHOU_CONTROLLED
        elif isinstance(sampler_instance, SamplerZouGrid):
            return SamplerType.ZOU_GRID
        else:
            raise ValueError(f'Unknown samples class: {sampler_instance.__class__.__name__}')
