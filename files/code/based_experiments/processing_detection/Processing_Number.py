# Refactored Processing_Number.py
# - Added English comments and docstrings
# - Cleaned repeated code paths and improved function names
# - Kept behavior compatible with original script (McCatch calling external JAR)
# - Added safer delimiter detection and clearer error handling

import os
import sys
import csv
import pandas as pd
import numpy as np
import multiprocessing.pool
import functools
import warnings
from typing import List, Tuple, Any, Optional, Dict

warnings.filterwarnings("ignore")

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.abod import ABOD
from pyod.models.deep_svdd import DeepSVDD

# --- Configuration ---

# List of dataset repository roots to process (relative paths expected)
datasets = [
    r'../../../datasets/experiments/finance/processed/number',
    r'../../../datasets/experiments/medicine/processed/number',
    r'../../../datasets/experiments/network_security/processed/number',
    r'../../../datasets/experiments/not_grouped/processed/number',
    r'../../../datasets/experiments/sciency/processed/number',
    r'../../../datasets/experiments/synthetic/processed/number',
]

# Parameter grid used in experiments (kept from original script)
params = {
    'KNN': [1, 3, 5, 7, 10, 15, 20, 25, 30, 35],
    'LOF': [1, 3, 5, 7, 10, 15, 20, 25, 30, 35],
    'ABOD': [1, 3, 5, 7, 10, 15, 20, 25, 30, 35],
    'iForest': list(range(1, 11)),
    'DeepSVDD': list(range(1, 11)),
    'McCatch': [15, 0.1, 0.1],
}

# Model metadata: maps model name to constructor class and available values
models: Dict[str, Dict[str, Any]] = {
    'KNN': {'alg': KNN, 'values': params['KNN']},
    'LOF': {'alg': LOF, 'values': params['LOF']},
    'iForest': {'alg': IForest, 'values': params['iForest']},
    'ABOD': {'alg': ABOD, 'values': params['ABOD']},
    'DeepSVDD': {'alg': DeepSVDD, 'values': params['DeepSVDD']},
    # McCatch handled separately (external tool)
    'McCatch': {'alg': None, 'values': params['McCatch']},
}


# --- Utility functions ---

def timeout(max_seconds: float):
    """
    Decorator to time-limit execution of a function using a thread pool.
    Raises multiprocessing.TimeoutError when exceeded.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(func, args, kwargs)
            return async_result.get(max_seconds)
        return wrapper
    return decorator


def detect_delimiter(file_path: str) -> str:
    """
    Detect CSV delimiter using csv.Sniffer, defaults to comma on failure.
    """
    try:
        with open(file_path, 'r', newline='') as f:
            sample = f.read(2048)
            if not sample:
                return ','
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
    except Exception:
        # Fallback delimiter
        return ','


def ranking_numeric(values: List[float], ascending: bool = False) -> List[int]:
    """
    Generate a ranking list for numeric values.
    Higher values get better ranks unless ascending=True.
    Returns list of ranks aligned with original order.
    """
    indexed = list(enumerate(values))
    # sort by value; reverse when descending desired
    sorted_indexed = sorted(indexed, key=lambda x: x[1], reverse=not ascending)
    rank_map = {orig_idx: rank + 1 for rank, (orig_idx, _) in enumerate(sorted_indexed)}
    return [rank_map[i] for i in range(len(values))]


def top_n_indices(values: List[float], n: int, descending: bool = True) -> List[int]:
    """
    Return indices of the top-n values from 'values'.
    If n > len(values) will return all indices sorted.
    """
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=descending)
    return sorted_indices[:max(0, min(n, len(values)))]


def detection_matches_ground_truth(ground_truth: pd.Series, predicted_indices: List[int]) -> List[bool]:
    """
    Compare predicted outlier indices with ground truth labels ('yes'/'no').
    Returns a boolean list indicating whether each point was correctly labeled as outlier/normal.
    """
    matches = []
    predicted_set = set(predicted_indices)
    for idx, label in enumerate(ground_truth):
        if idx in predicted_set:
            # predicted as outlier => correct if ground truth is 'yes'
            matches.append(label == 'yes')
        else:
            # predicted as normal => correct if ground truth is 'no'
            matches.append(label == 'no')
    return matches


def save_result(df: pd.DataFrame, base_path: str, dataset_filename: str, converter: str):
    """
    Append or write results DataFrame to CSV under results/number/<model>/<converter>/<dataset_filename>.
    Keeps consistent column ordering.
    """
    columns_export = ['dataset', 'algorithm', 'parameter', 'point', 'type', 'detect', 'score', 'ranking']
    out_dir = os.path.join(base_path, converter)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, dataset_filename)
    if os.path.exists(out_file):
        df[columns_export].to_csv(out_file, sep=';', index=False, mode='a', header=False)
    else:
        df[columns_export].to_csv(out_file, sep=';', index=False)


def load_dataset(full_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load a CSV dataset, detect delimiter, and prepare columns used by processing.
    Returns: df (with added metadata columns), X (features), Y (outlier labels)
    """
    df = pd.read_csv(full_path, sep=detect_delimiter(full_path))
    if 'outlier' not in df.columns:
        raise ValueError(f"Missing 'outlier' column in dataset: {full_path}")
    X = df.drop(columns=['outlier'])
    Y = df['outlier']
    dataset_name = os.path.basename(full_path)
    df['dataset'] = dataset_name
    df['parameter'] = ''
    df['point'] = list(range(1, len(df) + 1))
    # Type 'I' for inlier ('no'), 'O' for outlier ('yes')
    df['type'] = ['I' if v == 'no' else 'O' for v in Y]
    df['detect'] = ''
    return df, X, Y


def merge_dataframe(df_final: pd.DataFrame, df_partial: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate two DataFrames. If the final is empty, return a copy of the partial.
    """
    if df_final is None or df_final.empty:
        return df_partial.copy()
    return pd.concat([df_final, df_partial.copy()], ignore_index=True)


# --- Model instantiation and execution ---

@timeout(10800.0)  # 3 hours timeout for a single model run
def create_and_fit_instance(model_name: str, cls: Any, param_value: Any, data: pd.DataFrame, dataset_name: str):
    """
    Instantiate model class 'cls' with parameters based on model_name and param_value.
    Fit the model on 'data' and return the fitted instance.
    This wrapper is time-limited by the timeout decorator.
    """
    try:
        if model_name == 'KNN':
            inst = cls(n_neighbors=int(param_value))
            inst.fit(data)
        elif model_name == 'LOF':
            inst = cls(n_neighbors=int(param_value))
            inst.fit(data)
        elif model_name == 'ABOD':
            # ABOD supports n_neighbors and method
            inst = cls(n_neighbors=int(param_value), method='fast')
            inst.fit(data)
        elif model_name == 'iForest':
            # original script did not pass params for iForest
            inst = cls()
            inst.fit(data)
        elif model_name == 'DeepSVDD':
            inst = cls(verbose=0)
            inst.fit(data)
        else:
            # Unknown or external model handled elsewhere
            inst = None
        return inst
    except Exception as exc:
        # Return None on error (including timeout)
        print(f"ERROR creating/fitting {model_name} param={param_value} dataset={dataset_name}: {exc}")
        return None


def process_model(repository: str, dataset_filename: str, converter: str, model_name: str, param_value: Any):
    """
    Main processing function called per model/parameter combination.
    It loads dataset, fits the model, computes scores and detection indicators, and saves results.
    """
    # Prepare result path root
    path_result = os.path.join(repository, 'number', model_name, converter)
    os.makedirs(path_result, exist_ok=True)
    dataset_path = os.path.join(path_result, dataset_filename)
    try:
        df_tmp, X, Y = load_dataset(dataset_path)
    except Exception as e:
        print(f"Failed to load dataset {dataset_path}: {e}")
        return

    # number of true outliers in the dataset (ground truth)
    try:
        num_outliers = int(Y.value_counts().get('yes', 0))
    except Exception:
        num_outliers = 0

    df_tmp['algorithm'] = model_name
    df_all: Optional[pd.DataFrame] = pd.DataFrame()

    # Handle external McCatch separately (calls Java JAR)
    if model_name == 'McCatch':
        dbc = dataset_path
        output_file = 'output.txt'
        if os.path.exists(output_file):
            os.remove(output_file)
        try:
            mccatch_jar = os.path.normpath(r'..\..\algorithms\McCatch\target\mccatch-1.0.jar')
            # Execute McCatch and redirect stdout to output.txt
            command = f'java -jar "{mccatch_jar}" -algorithm elki.outlier.distance.McCatch -time -dbc.in "{dbc}" > {output_file}'
            os.system(command)

            df_tmp['parameter'] = ','.join(map(str, params['McCatch']))
            # Parse output file lines that start with "ID="
            with open(output_file, 'r') as fo:
                lines = fo.read().splitlines()

            result_rows = []
            included = set()
            for line in lines:
                if line.startswith('ID='):
                    # Expect format like: ID=<id> ... <score>
                    cleaned = line.replace('McCatch score per point=', '').replace('ID=', '')
                    parts = cleaned.split()
                    if parts and parts[0] not in included:
                        result_rows.append(parts)
                        included.add(parts[0])

            if not result_rows:
                raise RuntimeError("McCatch produced no valid lines")

            df_mccatch = pd.DataFrame(result_rows)
            score_col = df_mccatch.columns[-1]
            df_mccatch[score_col] = df_mccatch[score_col].astype(float)
            scores = df_mccatch[score_col].tolist()
            y_pred_indices = top_n_indices(scores, n=num_outliers)
            df_tmp['score'] = scores
            df_tmp['ranking'] = ranking_numeric(scores, ascending=False)
            df_tmp['detect'] = detection_matches_ground_truth(Y, y_pred_indices)
            df_all = merge_dataframe(df_all, df_tmp)
        except Exception as exc:
            print(f"McCatch processing failed for {dbc}: {exc}")
    else:
        # Standard PyOD model flow
        cls = models.get(model_name, {}).get('alg')
        if cls is None:
            print(f"No constructor for model {model_name}; skipping.")
            return

        df_tmp['parameter'] = str(param_value)

        instance = create_and_fit_instance(model_name, cls, param_value, X, dataset_filename)
        if instance is not None:
            # Attempt to obtain decision scores from the instance
            scores = None
            try:
                scores = instance.decision_scores_
            except Exception:
                # Some PyOD objects may expose negative_outlier_factor_ (not standard)
                try:
                    scores = instance.negative_outlier_factor_
                except Exception:
                    scores = None

            if scores is None:
                # If no scores available, mark as zeros and false detections
                df_tmp['score'] = 0.0
                df_tmp['ranking'] = 0
                df_tmp['detect'] = False
            else:
                # Compute predicted outlier indices based on top-n scores and compute metrics
                scores_list = list(scores)
                y_pred_indices = top_n_indices(scores_list, n=num_outliers)
                df_tmp['score'] = scores_list
                df_tmp['ranking'] = ranking_numeric(scores_list, ascending=False)
                df_tmp['detect'] = detection_matches_ground_truth(Y, y_pred_indices)
        else:
            # Model instantiation or fitting failed
            df_tmp['score'] = 0.0
            df_tmp['ranking'] = 0
            df_tmp['detect'] = False

        df_all = merge_dataframe(df_all, df_tmp)

    # Save results to disk
    try:
        save_result(df_all, os.path.join('results', 'number', model_name), dataset_filename, converter)
    except Exception as exc:
        print(f"Failed to save results for {model_name} {dataset_filename}: {exc}")


# --- CLI entrypoint ---

if __name__ == "__main__":
    """
    Expected CLI arguments:
      1) repository (root folder)
      2) dataset filename (e.g., mydata.csv)
      3) converter (subfolder name)
      4) model (e.g., KNN, LOF, iForest, ABOD, DeepSVDD, McCatch)
      5) parameter (value for the model, string)
    Example:
      python Processing_Number.py ../../../datasets ... mydata.csv KNN 5
    """
    if len(sys.argv) < 6:
        print("Usage: Processing_Number.py <repository> <dataset> <converter> <model> <param>")
        sys.exit(1)

    repository_arg = sys.argv[1]
    dataset_arg = sys.argv[2]
    converter_arg = sys.argv[3]
    model_arg = sys.argv[4]
    param_arg = sys.argv[5]

    try:
        process_model(repository_arg, dataset_arg, converter_arg, model_arg, param_arg)
    except Exception as e:
        print(f"ERROR processing {repository_arg} {dataset_arg} {converter_arg} {model_arg} {param_arg}: {e}")
    finally:
        # Create a finished marker (original script echoed into finish)
        try:
            with open('finish', 'w') as f:
                f.write('')
        except Exception:
            pass