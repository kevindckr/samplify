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

- GRSS18: https://machinelearning.ee.uh.edu/2018-ieee-grss-data-fusion-challenge-fusion-of-multispectral-lidar-and-hyperspectral-data/
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

	@article{Decker2025,
		abstract = {Identified as early as 2000, the challenges involved in developing and assessing remote sensing models with small datasets remain, with one key issue persisting: the misuse of random sampling to generate training and testing data. This practice often introduces a high degree of correlation between the sets, leading to an overestimation of model generalizability. Despite the early recognition of this problem, few researchers have investigated its nuances or developed effective sampling techniques to address it. Our survey highlights that mitigation strategies to reduce this bias remain underutilized in practice, distorting the interpretation and comparison of results across the field. In this work, we introduce a set of desirable characteristics to evaluate sampling algorithms, with a primary focus on their tendency to induce correlation between training and test data, while also accounting for other relevant factors. Using these characteristics, we survey 146 articles, identify 16 unique sampling algorithms, and evaluate them. Our evaluation reveals two broad archetypes of sampling techniques that effectively mitigate correlation and are suitable for model development.},
		author = {Kevin T Decker and Brett J Borghetti},
		doi = {10.3390/RS17081373},
		issn = {2072-4292},
		issue = {8},
		journal = {Remote Sensing 2025, Vol. 17, Page 1373},
		keywords = {correlation,generalization,model assessment,remote sensing,sampling algorithm},
		month = {4},
		pages = {1373},
		publisher = {Multidisciplinary Digital Publishing Institute},
		title = {A Survey of Sampling Methods for Hyperspectral Remote Sensing: Addressing Bias Induced by Random Sampling},
		volume = {17},
		url = {https://www.mdpi.com/2072-4292/17/8/1373/htm https://www.mdpi.com/2072-4292/17/8/1373},
		year = {2025}
	}
