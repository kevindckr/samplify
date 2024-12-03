import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple

from empirical.sampler_characterizer import SamplerCharacterizer


class SamplerExperimentResults:
    def __init__(self) -> None:
        # Runtime of experiment in seconds
        self.times: List[float] = []

        # Overlap Percentage - "Mutually Exclusive Subset Assignment"
        self.overlap_percentage: List[float] = []

        # Global Moran's I - "Spatial Autocorrelation"
        self.morans: List[float] = []

        # KL Divergence - "Commensurate Class Distributions" (Ignore unlabeled class)
        self.kl_divergence_train: List[float] = []
        self.kl_divergence_test: List[float] = []

        # Different Ratio - "Bernoulli Distribution based Allocation"
        self.difference_ratio: List[float] = []

        # ---- Other Information ----

        # Number of training and testing cpls
        self.num_initial_valid_cpls: List[int] = []
        self.num_train_cpls: List[int] = []
        self.num_test_cpls: List[int] = []

        # Number of available, used, and unused cpls
        self.num_used_cpls: List[int] = []
        self.num_unused_cpls: List[int] = []

        # Number of cpls that are unlabelled
        self.num_unlabelled_train: List[int] = []
        self.num_unlabelled_test: List[int] = []


    def append_run_results(self, total_time: float, characterizer: SamplerCharacterizer):
        # Append runtime
        self.times.append(total_time)

        # Overlap Percentage - "Mutually Exclusive Subset Assignment"
        self.overlap_percentage.append(characterizer.overlap_percentage)

        # Global Moran's I - "Spatial Autocorrelation"
        self.morans.append(characterizer.morans)

        # KL Divergence - "Commensurate Class Distributions" (Ignore unlabeled class)
        self.kl_divergence_train.append(characterizer.kl_divergence_train)
        self.kl_divergence_test.append(characterizer.kl_divergence_test)

        # Different Ratio - "Bernoulli Distribution based Allocation"
        self.difference_ratio.append(characterizer.difference_ratio)

        # ---- Other Information ----

        # Number of training and testing cpls
        self.num_initial_valid_cpls.append(characterizer.num_initial_valid_cpls)
        self.num_train_cpls.append(characterizer.num_train_cpls)
        self.num_test_cpls.append(characterizer.num_test_cpls)

        # Number of available, used, and unused cpls
        self.num_used_cpls.append(characterizer.num_used_cpls)
        self.num_unused_cpls.append(characterizer.num_unused_cpls)

        # Number of cpls that are unlabelled
        self.num_unlabelled_train.append(characterizer.num_unlabelled_train)
        self.num_unlabelled_test.append(characterizer.num_unlabelled_test)

    
    def serialize_run(self, run_idx: int, result_dir: str, print_results: bool = True):
        run_json = {
            'run_idx': run_idx,
            'total_time': self.times[run_idx],

            # Overlap Percentage - "Mutually Exclusive Subset Assignment"
            'overlap_percentage': self.overlap_percentage[run_idx],

            # Global Moran's I - "Spatial Autocorrelation"
            'morans': self.morans[run_idx],

            # KL Divergence - "Commensurate Class Distributions" (Ignore unlabeled class)
            'kl_divergence_train': self.kl_divergence_train[run_idx],
            'kl_divergence_test': self.kl_divergence_test[run_idx],

            # Different Ratio - "Bernoulli Distribution based Allocation"
            'difference_ratio': self.difference_ratio[run_idx],

            # ---- Other Information ----

            # Number of training and testing cpls
            'num_initial_valid_cpls': self.num_initial_valid_cpls[run_idx],
            'num_train_cpls': self.num_train_cpls[run_idx],
            'num_test_cpls': self.num_test_cpls[run_idx],

            # Number of available, used, and unused cpls
            'num_used_cpls': self.num_used_cpls[run_idx],
            'num_unused_cpls': self.num_unused_cpls[run_idx],

            # Number of cpls that are unlabelled
            'num_unlabelled_train': self.num_unlabelled_train[run_idx],
            'num_unlabelled_test': self.num_unlabelled_test[run_idx],
        }

        if print_results:
            print(f'> Run Results:\n{json.dumps(run_json, indent=4)}')
        
        with open(os.path.join(result_dir, f'experiment_result_r{run_idx}.json'), 'w') as f:
            json.dump(run_json, f, indent=4)
    
    def serialize_averages(self, result_dir: str, print_results: bool = True):
        avg_json = {
            'total_runs': len(self.times),
            # Averages
            'avg_total_time': np.average(self.times),

            # Overlap Percentage - "Mutually Exclusive Subset Assignment"
            'avg_overlap_percentage': np.average(self.overlap_percentage),

            # Global Moran's I - "Spatial Autocorrelation"
            'avg_morans': np.average(self.morans),

            # KL Divergence - "Commensurate Class Distributions" (Ignore unlabeled class)
            'avg_kl_divergence_train': np.average(self.kl_divergence_train),
            'avg_kl_divergence_test': np.average(self.kl_divergence_test),

            # Different Ratio - "Bernoulli Distribution based Allocation"
            'avg_difference_ratio': np.average(self.difference_ratio),

            # ---- Other Information ----

            # Number of training and testing cpls
            'avg_num_initial_valid_cpls': np.average(self.num_initial_valid_cpls),
            'avg_num_train_cpls': np.average(self.num_train_cpls),
            'avg_num_test_cpls': np.average(self.num_test_cpls),

            # Number of available, used, and unused cpls
            'avg_num_used_cpls': np.average(self.num_used_cpls),
            'avg_num_unused_cpls': np.average(self.num_unused_cpls),

            # Number of cpls that are unlabelled
            'avg_num_unlabelled_train': np.average(self.num_unlabelled_train),
            'avg_num_unlabelled_test': np.average(self.num_unlabelled_test),
        }

        if print_results:
            print(f'> Final Average Results:\n{json.dumps(avg_json, indent=4)}')
        
        with open(os.path.join(result_dir, 'final_experiment_result.json'), 'w') as f:
            json.dump(avg_json, f, indent=4)
