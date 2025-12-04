# Refactored Executor.py
# - English comments and docstrings
# - Clearer process orchestration and robust delimiter detection
# - Uses polling on a 'finish' marker file produced by worker scripts

import os
import time
import subprocess
import csv
import numpy as np
import pandas as pd
from typing import List

# List of dataset repositories to iterate
datasets = [
    r'../../../datasets/base_experiments/finance/processed/number',
    r'../../../datasets/base_experiments/medicine/processed/number',
    r'../../../datasets/base_experiments/network_security/processed/number',
    r'../../../datasets/base_experiments/not_grouped/processed/number',
    r'../../../datasets/base_experiments/sciency/processed/number',
    r'../../../datasets/base_experiments/synthetic/processed/number',
]

models_to_run = ['KNN', 'LOF', 'iForest', 'DeepSVDD', 'McCatch', 'ABOD',]

# Parameter templates used by algorithms
params = {
    'KNN': [1, 3, 5, 7, 10, 15, 20, 25, 30, 35],
    'LOF': [1, 3, 5, 7, 10, 15, 20, 25, 30, 35],
    'iForest': list(range(1, 11)),
    'DeepSVDD': list(range(1, 11)),
    'McCatch': ['15,0.1,0.1'],
    'ABOD': [1, 3, 5, 7, 10, 15, 20, 25, 30, 35],
}


def detect_delimiter(file_path: str) -> str:
    """
    Detect CSV delimiter using csv.Sniffer.
    Defaults to comma on failure.
    """
    try:
        with open(file_path, 'r', newline='') as fh:
            sample = fh.read(2048)
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
    except Exception:
        return ','


def wait_for_finish_marker(poll_interval: float = 5.0, timeout: float = 3600.0) -> bool:
    """
    Poll for the presence of the 'finish' marker file created by worker script.
    Returns True when marker is found (and removes it). Returns False on timeout.
    """
    start = time.time()
    marker = 'finish'
    while True:
        if os.path.exists(marker):
            try:
                os.remove(marker)
            except Exception:
                pass
            return True
        if (time.time() - start) > timeout:
            return False
        time.sleep(poll_interval)


def generate_param_values(minimum: int, maximum: int, num_param: int) -> List[int]:
    """
    Generate evenly spaced integer parameter values between minimum and maximum.
    """
    return [int(n) for n in list(np.linspace(minimum, maximum, num_param))]


# Main executor loop
for repository in datasets:
    number_dir = os.path.join(repository, 'number')
    if not os.path.isdir(number_dir):
        print(f"Repository number dir not found, skipping: {number_dir}")
        continue

    # You can restrict converters here; default reads all subfolders under 'number'
    converters = os.listdir(number_dir)

    for converter in converters:
        converter_dir = os.path.join(number_dir, converter)
        if not os.path.isdir(converter_dir):
            print(f"Converter directory not found, skipping: {converter_dir}")
            continue

        files = os.listdir(converter_dir)
        for dataset in files:
            # skip directories and non-files
            dataset_path = os.path.join(converter_dir, dataset)
            if not os.path.isfile(dataset_path):
                continue

            print(f'Preparing dataset: {dataset_path}')
            # Read dataset to determine length (used to compute parameter ranges)
            try:
                delim = detect_delimiter(dataset_path)
                df = pd.read_csv(dataset_path, sep=delim)
            except Exception as e:
                print(f"Failed to read dataset {dataset_path}: {e}")
                continue

            for model in models_to_run:

                print(f'Algorithm: {model} - Parameters: {params[model]}')

                for param in params[model]:
                    # Skip known avoid combinations if configured (optional)
                    # Check if result already exists for this dataset/param
                    path_results = os.path.join('results', 'number', model, converter, dataset)
                    if os.path.exists(path_results):
                        try:
                            result = pd.read_csv(path_results, sep=';')
                            if len(result.query("parameter == @param")) > 0:
                                print(f"Result exists for param {param}, skipping.")
                                continue
                        except Exception:
                            pass

                    # Build command to run the worker script (Processing_Number.py)
                    python_exe = os.path.join('venv', 'Scripts', 'python.exe')
                    script = 'Processing_Number.py'
                    cmd = [python_exe, script, repository, dataset, converter, model, str(param)]

                    print('Launching:', ' '.join(cmd))

                    # Ensure any stale finish marker removed before launching
                    if os.path.exists('finish'):
                        try:
                            os.remove('finish')
                        except Exception:
                            pass

                    # Start worker (non-blocking) and then wait for its finish marker
                    try:
                        subprocess.Popen(cmd)
                    except Exception as e:
                        print(f"Failed to start worker for {dataset} param {param}: {e}")
                        continue

                    # Wait until worker creates the 'finish' marker (or timeout)
                    if not wait_for_finish_marker(poll_interval=10.0, timeout=3 * 3600):
                        print(f"Worker timed out for {dataset} param {param}, moving on.")
                        # Optionally attempt to kill or cleanup here
                        continue

                    print(f"Completed: {dataset} param {param}")

# End executor loop
print("Executor finished all tasks.")