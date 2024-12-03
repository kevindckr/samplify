import os
import sys
import time
import traceback
from io import TextIOWrapper
from datetime import datetime
from typing import List, TextIO

from datasets._dataset_enum import DatasetType
from datasets._dataset_base import BaseImageDataset

from samplers._sampler_enum import SamplerType
from samplers._sampler_base import BaseImageSampler

from empirical.sampler_characterizer import SamplerCharacterizer
from empirical.sampler_experiment_results import SamplerExperimentResults


class SamplifySamplerExperiment:
    def __init__(self, dataset: BaseImageDataset, sampler: BaseImageSampler, base_dir: str, log_stdout: bool = True, total_runs: int = 3) -> None:
        self.dataset: BaseImageDataset = dataset
        self.sampler: BaseImageSampler = sampler
        self.base_dir: str = base_dir
        self.log_stdout: bool = log_stdout
        self.total_runs: int = total_runs

        # Derived from base_dir
        self.result_dir: str = None
        self.create_result_dir_name()

        # Used by _capture_stdout to capture and replace stdout during training
        self.sys_stdout: TextIO = None
        self.stdout_logfile: TextIOWrapper = None
    
    def create_result_dir_name(self):
        """ Generate the name of the result directory to use """
        # Generate the name for the final subdir for this experiment
        ds_type = DatasetType.get_dataset_type(self.dataset)
        sp_type = SamplerType.get_sampler_type(self.sampler)
        self.result_dir = self.result_dir_name(
            self.base_dir, ds_type, sp_type, self.sampler.train_ratio, 
            self.sampler.patch_size, self.total_runs)
    
    @staticmethod
    def result_dir_name(base_dir, dataset_type, sampler_type, train_ratio, patch_size, total_runs):
        # Generate the name for the final subdir for this experiment
        trn_str = f'{int(train_ratio * 100)}'
        pat_str = f'{patch_size[0]}x{patch_size[1]}'
        run_str = f'R{total_runs}'
        last_dir_str = f'{pat_str}_{trn_str}_{run_str}'

        # Create the result_dir name and directory
        return os.path.join(base_dir, f'{dataset_type.name}',  f'{sampler_type.name}', last_dir_str)

    
    def print_information(self):
        """ Print information about this experiment to stdout """
        print(f'> Experiment Overview')
        print(f'  Start Time: {datetime.now().strftime("%m/%d/%y %H:%M:%S.%f")}')
        print(f'  Dataset: {self.dataset.name}')
        print(f'  Sampler: {self.sampler.name}')
        print(f'  Total Runs: {self.total_runs}')
        print(f'  Train Ratio: {self.sampler.train_ratio}')
        print(f'  Patch Size: {self.sampler.patch_size}')
    
    def run_experiment(self):
        try:
            # Create the result directory itself
            os.makedirs(self.result_dir, exist_ok=True)

            # Capture stdout and write to file
            if self.log_stdout: self._capture_stdout()

            # Print information about the experiment so it sits at the top of the stdout.log
            self.print_information()

            # Accumulate results and then serialize the averages
            experiment_results = SamplerExperimentResults()

            # Perform each of the runs
            for run_idx in range(self.total_runs):
                print(f'> Starting run: {run_idx}/{self.total_runs}')

                # Perform sampling and save results to file
                print(f'> Sampling...')
                self.sampler.random_seed += 1

                start_time = time.time()
                self.sampler.sample()
                total_time = (time.time() - start_time)

                # Serialize the sampler object to a file
                self.sampler.save(os.path.join(self.result_dir, f'sampler_result_r{run_idx}.json'))

                # Characterize the sampler
                sampler_characterizer = SamplerCharacterizer(self.sampler)
                sampler_characterizer.characterize_sampler()

                # Print and write the final results
                experiment_results.append_run_results(total_time, sampler_characterizer)
                experiment_results.serialize_run(run_idx, self.result_dir)

            # Average across runs, print, and serialize
            experiment_results.serialize_averages(self.result_dir)

            return True
        
        # Just write a file to denote that the experiment was interruptted
        except KeyboardInterrupt as e:
            print(f'! Keyboard Interrupt !')
            with open(os.path.join(self.result_dir, 'interrupted.log'), 'a') as f:
                f.write(f'Keyboard Interrupt')
            return False
        
        # Write the error message and traceback to the log file
        except Exception as e:
            print(f'! Experiment Error: {e}')
            with open(os.path.join(self.result_dir, 'failed.log'), 'a') as f:
                f.write(f'Experiment Error: {e}\n\n\nTrackback:\n')
                traceback.print_exc(file=f)
            return False
        
        # Release stdout and close file
        finally:
            if self.log_stdout: self._release_stdout()

    def _capture_stdout(self):
        # Create a file to pipe stdout of the process this experiment is running in to
        stdout_path = os.path.join(self.result_dir, f'stdout.log')

        # Capture stdout and direct to the log_file
        self.sys_stdout: TextIO = sys.stdout
        self.stdout_logfile: TextIOWrapper = open(stdout_path, 'w', buffering=1)
        sys.stdout = self.stdout_logfile 

    def _release_stdout(self):
        # Release stdout back to sys and close the stdout log file
        if self.sys_stdout: sys.stdout = self.sys_stdout
        if self.stdout_logfile: self.stdout_logfile.close()
