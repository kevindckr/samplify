import os
import sys
import time
import psutil
import argparse
from tqdm import tqdm
import concurrent.futures

from paths import *

from plotting.dataset.common import *

from datasets._dataset_enum import DatasetType
from samplers._sampler_enum import SamplerType

from empirical.sampler_experiment import SamplifySamplerExperiment


def get_cpu_usage():
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    used_percent = memory.percent
    return used_gb, used_percent


def determine_experiment_completed(experiment: SamplifySamplerExperiment):
    result_dir = experiment.result_dir
    exp_exists = os.path.exists(result_dir)
    if exp_exists:
        # Check for various conditions that require directory deletion
        incomplete_indicators = ['interrupted.log', 'failed.log']
        needs_cleanup = any(os.path.exists(os.path.join(result_dir, log)) for log in incomplete_indicators)
        partial_result = not os.path.exists(os.path.join(result_dir, 'final_experiment_result.json'))

        if needs_cleanup or partial_result:
            # Attempt to delete the directory if any indicators of incomplete experiments are found
            try:
                #shutil.rmtree(result_dir)
                print(f'Deleted incomplete experiment directory: {result_dir}')
                print(f'  Needs Cleanup: {needs_cleanup} Partial Result: {partial_result}')
                return False  # The experiment needs to be rerun
            except Exception as e:
                print(f'! Could not delete incomplete experiment directory: {result_dir}\n{e}')
                return None
        # If the directory exists and no cleanup was needed, the experiment is considered complete
        return True
    # If the directory does not exist, the experiment needs to be run
    return False


def run_experiment_wrapper(experiment: SamplifySamplerExperiment):
    experiment.run_experiment()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run experiments with different datasets.")
    parser.add_argument('dataset', type=str, choices=[d.name for d in DatasetType])
    args = parser.parse_args()
    try:
        datasets = [DatasetType[args.dataset]]
    except:
        print(f'Could not parse dataset type, defaulting to Salinas.')
        datasets = [DatasetType.Salinas]

    # Paths
    output_dir = os.path.join(OUTPUT_DIR, '_sampler_experiments')
    ensure_dir(output_dir)

    # Administartive/static parameters
    random_seed = 2813308004

    # The experiment parameters to iterate over
    repeats = 3
    train_ratios = [0.05, 0.10, 0.25]
    patch_sizes = [(5, 5), (9, 9), (15, 15)]
    #datasets = [dataset_type for dataset_type in DatasetType]
    datasets = [
        #DatasetType.Salinas
        #DatasetType.GRSS13,
        #DatasetType.GRSS18,
        #DatasetType.Botswana,
        #DatasetType.IndianPines,
        #DatasetType.KennedySpaceCenter,
        #DatasetType.PaviaCenter,
        DatasetType.PaviaUniversity,
    ]

    samplers = [
        # Random types
        SamplerType.CLARK_STRATIFIED,
        SamplerType.LIU_RANDOM,
        SamplerType.RANDOM_EQUAL_COUNT,
        SamplerType.RANDOM_STRATIFIED,
        SamplerType.RANDOM_UNIFORM,
        # Controlled types
        SamplerType.ACQUARELLI_CONTROLLED,
        SamplerType.HANSCH_CLUSTER,
        SamplerType.LANGE_CONTROLLED,
        SamplerType.LIANG_CONTROLLED,
        SamplerType.ZHOU_CONTROLLED,
        # Grid types
        SamplerType.GRID,
        SamplerType.ZHANG_GRID,
        SamplerType.ZOU_GRID,
        # Spatial Partitioning
        SamplerType.DECKER_SPATIAL,
        # SamplerType.SPATIAL_SIMPLE,
        # Special types
        # SamplerType.DECKER_DYNAMIC,
        # SamplerType.GRSS13_SUGGESTED,
    ]

    # Instantiate all of the experiments to run
    print(f'> Instantiating Experiments...')
    start = time.time()

    from tqdm import tqdm

    # Instantiate all of the experiments to run
    experiments = []
    total_possible_experiments, total_previously_completed_experiments = 0, 0

    # Calculate total number of iterations for the progress bar
    total_iterations = len(datasets) * len(samplers) * len(patch_sizes) * len(train_ratios)

    # Initialize tqdm progress bar
    with tqdm(total=total_iterations, desc='Setting up experiments', unit='experiment') as pbar:
        for dataset_type in datasets:
            # Instantiate and pre-load the datasets, SamplifyExperiments share a common instance
            dataset_cls = DatasetType.get_dataset_class(dataset_type)
            dataset = dataset_cls(DATASET_DATA_DIR)
            dataset.load()
            #dataset.minmax_normalization()

            for sampler_type in samplers:
                sampler_cls = SamplerType.get_sampler_class(sampler_type)

                # Do not make experiments using the GRSS13_SUGGESTED sampler on a non-GRSS13 dataset
                if dataset_type != DatasetType.GRSS13 and sampler_type == SamplerType.GRSS13_SUGGESTED:
                    # Increment the progress bar for each skipped combination
                    pbar.update(len(patch_sizes) * len(train_ratios))
                    continue

                # Create the sampler's kwargs
                sampler_kwargs = {'dataset': dataset}

                # Create special kwargs for the GRSS13_SUGGESTED sampler on the GRSS13 dataset
                if dataset_type == DatasetType.GRSS13 and sampler_type == SamplerType.GRSS13_SUGGESTED:
                    sampler_kwargs = {'dataset': dataset, 'labels_train': dataset.labels_train, 'labels_test': dataset.labels_test}

                for patch_size in patch_sizes:
                    for train_ratio in train_ratios:
                        # Update the count of total possible experiments to report to user
                        total_possible_experiments += 1

                        # Instantiate the sampler and experiment. Sampling is performed in experiment process
                        sampler = sampler_cls(dataset.image, dataset.labels, dataset.classes, patch_size, 
                                            train_ratio, sampler_type.name, random_seed, **sampler_kwargs)
                        experiment = SamplifySamplerExperiment(dataset, sampler, output_dir, log_stdout=True, total_runs=repeats)

                        # Using uniquely named experiment result directory, determine its status
                        completed = determine_experiment_completed(experiment)

                        if completed is None:
                            raise RuntimeError(f'! Could not cleanup and experiment directory, requires intervention !')

                        if not completed:
                            experiments.append(experiment)
                        else:
                            total_previously_completed_experiments += 1  # Update count of completed experiments
                        
                        # Update the progress bar for each completed iteration
                        pbar.update(1)

    # Show the user this output and wait.
    print(f'Total Instantiation Time: {(time.time() - start):.2f}')
    print(f'Total Possible Experiments: {total_possible_experiments}')
    print(f'Previous Completed Experiments: {total_previously_completed_experiments}')
    print(f'Experiments to Run: {len(experiments)}')
    #input('\nPress any key to begin running experiments...')

    # Run all of the experiments
    max_concurrent_processes = 10#20
    completed_inds, running_inds, ready_inds = [], [], []
    experiment_status = {i: (e, False, None) for i, e in enumerate(experiments)}

    max_cpu_ram_gb, max_cpu_ram_per = get_cpu_usage()
    max_concurrent_processes = 10 

    # Use a ProcessPoolExecutor to manage the pool of worker processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_processes) as executor:
        # Submit all experiments to the executor
        future_to_experiment = {
            executor.submit(run_experiment_wrapper, exp): exp for exp in experiments
        }

        max_cpu_ram_gb, max_cpu_ram_per = get_cpu_usage()
        completed = 0
        total_experiments = len(experiments)

        # Initialize the status line
        status_template = (
            'CPU RAM : {cpu_ram_gb:.1f} GB ({cpu_ram_per:.1f}%) | '
            'Completed: {completed}/{total} | '
            'Running: {running}'
        )
        print('')  # Ensure there's at least one line to overwrite

        # As each experiment completes, print its status
        for future in concurrent.futures.as_completed(future_to_experiment):
            exp = future_to_experiment[future]
            try:
                # If your experiments return a result, you can retrieve it here
                result = future.result()
            except Exception as exc:
                print(f'\n  FAILED - {SamplerType.get_sampler_type(exp.sampler)} {exp.sampler.patch_size} {exp.sampler.train_ratio}\n')
            else:
                pass
                #print(f'Experiment completed.')
            completed += 1

            # Update and print CPU usage
            cpu_ram_gb, cpu_ram_per = get_cpu_usage()
            if cpu_ram_gb > max_cpu_ram_gb:
                max_cpu_ram_gb = cpu_ram_gb
                max_cpu_ram_per = cpu_ram_per

            running = max_concurrent_processes - (completed % max_concurrent_processes)
            status = status_template.format(
                cpu_ram_gb=cpu_ram_gb,
                cpu_ram_per=cpu_ram_per,
                completed=completed,
                total=total_experiments,
                running=running
            )

            # Overwrite the previous line with the new status
            sys.stdout.write('\r' + status)
            sys.stdout.flush()

    print('All experiments completed.')