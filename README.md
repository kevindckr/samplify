# Samplify

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
Samplify is a Python library designed to streamline data sampling processes for various experiments, particularly within the context of remote sensing and other scientific domains. The library provides tools for creating and manipulating datasets, sampling strategies, and visualizing results through convenient, modular components.

### Key Features
- 17 of the most commonly used sampling strategies for hyperspectral imagery (or remote sensing imagery in general)
- Empirical testing framework to validate sampling strategies
- Sampling results visualization using "Footprint" plots (See academic article for more details)
- Support for 8 common remote sensing datasets
- Extensible architecture for custom datasets and samplers

### Applications
- Training data generation for machine learning models
- Spatial sampling validation
- Remote sensing research

## Installation
To install Samplify, simply clone the repository and run the setup.py script:

    git clone https://github.com/kevindckr/samplify.git
    cd samplify
    python setup.py install

## Directory Structure
**datasets/:** Dataset base class and concrete implementations.

**samplers/:** Sampling base class and concrete implementations.

**plotting/:** Various simple plotting functions including Footprint Plots.

**empirical/:** Empirical sampler testing suite.

**paths.py:** Path management for the library.

## Data

Users are responsible for downloading supported dataset data. Original sources are:

- GRSS18: https://hyperspectral.ee.uh.edu/?page_id=1075
- GRSS13: https://figshare.com/articles/dataset/GRSS_HOS_MAT/16528845
- All Others: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

Though, these links may not work over time. If neccessary please send an email and the authors' copies can be provided.

## Usage

#### Instantiating an Existing Dataset
You can load one of the existing datasets from the library:

	from paths import DATASET_DATA_DIR
	from samplify.datasets.indian_pines import DatasetIndianPines
	dataset = DatasetIndianPines(DATASET_DATA_DIR)
	dataset.load()

Currently supported datasets are:

1. Botswana
2. GRSS13
3. GRSS18
4. Indian Pines
5. KSC Florida
6. Pavia Center
7. Pavia University
8. Salinas

Note: Only the hyperspectral modality for these datasets is loaded.

#### Creating a Dataset

To load a new dataset, subclass the [BaseImageDataset](samplify/datasets/_dataset_base.py) class and load as described.

#### Instantiating an Existing Sampler

You can load one of the existing datasets from the library:

	from paths import DATASET_DATA_DIR
	from samplify.datasets.indian_pines import DatasetIndianPines
	from samplify.samplers.hansch_cluster import SamplerHanschCluster
	# Instantiate a dataset
	dataset = DatasetIndianPines(DATASET_DATA_DIR)
	dataset.load()
	# Instantiate a Sampler
	kwargs = {'dataset': dataset}
	train_ratio, patch_size = 0.25, (9, 9)
	sampler = SamplerHanschCluster(dataset.image, dataset.labels, dataset.classes, patch_size, train_ratio, name='HanschCluster', **kwargs)
	# Run the sampler
	sampler.sample()

Currently supported samplers are:

1. AcquarelliControlled
2. ClarkStratified
3. DeckerDynamic (Not mentioned in academic article)
4. DeckerSpatial (Clustered Partitioning)
5. Grid
6. GRSS13Suggested
7. HanschCluster
8. LangeControlled
9. LiangControlled
10. LiuRandom
11. RandomEqualCount
12. RandomStratified
13. RandomUniform
14. SimpleSpatialPartitioning
15. ZhangGrid
16. ZhouControlled
17. ZouGrid

#### Creating a Sampler

To create a new sampler, subclass the [BaseImageSampler](samplify/samplers/_sampler_base.py) class and load as described.

## Contributing
Contributions to Samplify are welcome! Please feel free to submit a pull request or open an issue if you encounter any problems or have feature suggestions.

## License
This project is licensed under the MIT License.

## Citing Samplify
If you use the Samplify library in your research, please cite it as follows:

    @misc{samplify2024,
      title        = {Samplify: A Python Library for Sampling in Remote Sensing Datasets},
      author       = {Kevin T. Decker},
      year         = {2024},
      note         = {Software accompanying an unpublished research article},
      url          = {https://github.com/kevindckr/samplify},
      version      = {v0.1.0}
    }

This software accompanies a research article that is currently being prepared for publication. The repository will be updated with the appropriate citation details once the article is published.