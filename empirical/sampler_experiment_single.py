import os
import sys

from paths import *

from datasets._dataset_enum import DatasetType
from datasets._dataset_base import BaseImageDataset

from datasets.botswana import DatasetBotswana
from datasets.grss13 import DatasetGRSS13
from datasets.grss18 import DatasetGRSS18
from datasets.indian_pines import DatasetIndianPines
from datasets.kennedy_space_center import DatasetKennedySpaceCenter
from datasets.pavia_center import DatasetPaviaCenter
from datasets.pavia_university import DatasetPaviaUniversity
from datasets.salinas import DatasetSalinas

from samplers._sampler_enum import SamplerType
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

from empirical.sampler_experiment import SamplifySamplerExperiment


if __name__ == "__main__":
    # Paths
    output_dir = os.path.join(OUTPUT_DIR, '_sampler_experiments')
    ensure_dir(output_dir)

    # Experiment Parameters
    train_ratio = 0.15
    patch_size = (5, 5)
    random_seed = 2813308004

    # ==== Datasets ====
    # dataset = DatasetBotswana(DATASET_DATA_DIR)
    # dataset = DatasetGRSS13(DATASET_DATA_DIR)
    # dataset = DatasetGRSS18(DATASET_DATA_DIR)
    dataset = DatasetIndianPines(DATASET_DATA_DIR)
    # dataset = DatasetKennedySpaceCenter(DATASET_DATA_DIR)
    # dataset = DatasetPaviaCenter(DATASET_DATA_DIR)
    # dataset = DatasetPaviaUniversity(DATASET_DATA_DIR)
    # dataset = DatasetSalinas(DATASET_DATA_DIR)
    dataset.load()
    # dataset.bandmax_normalization()
    # dataset.minmax_normalization()

    # ==== Samplers ====
    kwargs = {'dataset': dataset}
    # kwargs = kwargs={'dataset': dataset, 'labels_train': dataset.labels_train, 'labels_test': dataset.labels_test}  # GRSS13 Suggested

    # sampler = SamplerAcquarelliControlled(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'AcquarelliControlled', random_seed, **kwargs)
    # sampler = SamplerClarkStratified(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'ClarkStratified', random_seed, **kwargs)
    # sampler = SamplerDeckerDynamic(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'DeckerDynamic', random_seed, **kwargs)
    # sampler = SamplerDeckerSpatial(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'DeckerSpatial', random_seed, **kwargs)
    # sampler = SamplerGrid(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'Grid', random_seed, **kwargs)
    # sampler = SamplerGRSS13Suggested(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'GRSS13Suggested', random_seed, **kwargs)
    sampler = SamplerHanschCluster(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'HanschCluster', random_seed, **kwargs)
    # sampler = SamplerLangeControlled(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'LangeControlled', random_seed, **kwargs)
    # sampler = SamplerLiangControlled(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'LiangControlled', random_seed, **kwargs)
    # sampler = SamplerLiuRandom(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'LiuRandom', random_seed, **kwargs)
    # sampler = SamplerRandomEqualCount(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'RandomEqualCount', random_seed, **kwargs)
    # sampler = SamplerRandomStratified(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'RandomStratified', random_seed, **kwargs)
    # sampler = SamplerRandomUniform(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'RandomUniform', random_seed, **kwargs)
    # sampler = SamplerSimpleSpatialPartitioning(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'SimpleSpatial', random_seed, **kwargs)
    # sampler = SamplerZhangGrid(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'ZhangGrid', random_seed, **kwargs)
    # sampler = SamplerZhouControlled(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'ZhouControlled', random_seed, **kwargs)
    # sampler = SamplerZouGrid(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, 'ZouGrid', random_seed, **kwargs)

    # ==== Experiment ====
    experiment = SamplifySamplerExperiment(dataset, sampler, output_dir, log_stdout=False)
    experiment.run_experiment()
