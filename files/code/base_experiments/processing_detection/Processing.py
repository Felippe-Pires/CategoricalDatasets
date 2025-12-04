# ...existing code...
import csv
import os
import pandas as pd
import warnings
from scipy.io import arff
from typing import List, Any, Dict

warnings.simplefilter("ignore")

from files.code.algorithms.python.CompreX.comprex import CompreX
from files.code.algorithms.python.AVF import AVF
from files.code.algorithms.python.EMAC.SCAN import SCAN
from coupled_biased_random_walks import CBRW
from files.code.algorithms.python.FPOF import FPOF

# List of algorithm configurations used in processing.
algorithms = [
    {'method': AVF, 'name': 'AVF', 'params': {'bins': [10, 50, 100]}},
    {'method': CBRW, 'name': 'CBRW', 'params': {'alpha': 0.95}},
    {'method': FPOF, 'name': 'FPOF', 'params': [{'minSupport': 0.1, 'mlen': 3}]},
    {'method': SCAN, 'name': 'SCAN', 'params': [{'dimensions': 128, 'alpha': 0.15}]},
    {'method': CompreX, 'name': 'CompreX', 'params': {}},
]

# Repositories to iterate (use list to preserve order).
repositories = [
    r'../../../datasets/base_experiments/finance/processed',
    r'../../../datasets/base_experiments/medicine/processed',
    r'../../../datasets/base_experiments/network_security/processed',
    r'../../../datasets/base_experiments/not_grouped/processed',
    r'../../../datasets/base_experiments/sciency/processed',
    r'../../../datasets/base_experiments/synthetic/processed',
]


def detect_delimiter(file_path: str) -> str:
    """
    Detect CSV delimiter using csv.Sniffer.
    Falls back to comma if detection fails or file is empty.
    """
    try:
        with open(file_path, 'r', newline='') as f:
            sample = f.read(2048)
            if not sample:
                return ','
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
    except Exception:
        return ','


def convert_numeric_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric values to strings so algorithms expecting categorical strings can process them.
    Non-numeric values are left unchanged.
    """
    return df.applymap(lambda x: str(x) if pd.api.types.is_number(x) else x)


def numeric_ranking(values: List[float], ascending: bool = True) -> List[int]:
    """
    Return ranking positions for a list of numeric values.
    If ascending=True, smaller values get better ranks (1 is best).
    Otherwise, larger values get better ranks.
    """
    indexed = list(enumerate(values))
    sorted_indexed = sorted(indexed, key=lambda x: x[1], reverse=not ascending)
    rank_map = {orig_idx: rank + 1 for rank, (orig_idx, _) in enumerate(sorted_indexed)}
    return [rank_map[i] for i in range(len(values))]


def top_n_indices(values: List[float], n: int, descending: bool = True) -> List[int]:
    """
    Return indices of the top-n values from 'values' list.
    If n is larger than the list length, return as many as available.
    """
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=descending)
    return sorted_indices[:max(0, min(n, len(values)))]


def detection_outlier(y_ground_truth: pd.Series, y_pred_indices: List[int]) -> List[bool]:
    """
    Given ground truth series of 'yes'/'no' and predicted outlier indices,
    return boolean list marking correct detection per point.
    """
    predicted_set = set(y_pred_indices)
    result = []
    for idx, true_val in enumerate(y_ground_truth):
        if idx in predicted_set:
            result.append(true_val == 'yes')
        else:
            result.append(true_val == 'no')
    return result


def save_result(df: pd.DataFrame, path: str, dataset: str) -> None:
    """
    Save processed results under path/dataset using semicolon separator.
    Creates directory if it doesn't exist.
    """
    columns_export = ['dataset', 'algorithm', 'parameter', 'point', 'type', 'detect', 'score', 'ranking']
    os.makedirs(path, exist_ok=True)
    out_file = os.path.join(path, dataset)
    df[columns_export].to_csv(out_file, sep=';', index=False)


# --- Main processing loop ---

for repository in repositories:
    # List datasets inside repository
    try:
        dataset_files = os.listdir(repository)
    except FileNotFoundError:
        print(f"Repository not found, skipping: {repository}")
        continue

    for dataset in dataset_files:
        dataset_path = os.path.join(repository, dataset)
        print(dataset)
        if not os.path.isfile(dataset_path):
            continue

        # Load dataset: support .arff and delimited text files
        if dataset_path.endswith('.arff'):
            try:
                data = arff.loadarff(dataset_path)
                df = pd.DataFrame(data[0])

                # Decode byte strings safely for object dtype columns
                for col in df.select_dtypes(include=['object']).columns:
                    # Only attempt decode when values are bytes
                    if df[col].apply(lambda x: isinstance(x, (bytes, bytearray))).any():
                        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else x)

                # Standardize outlier column to 'yes'/'no' with minority => 'yes'
                value_counts = df['outlier'].value_counts()
                labels = value_counts.index.tolist()
                if len(labels) >= 2:
                    if value_counts.iloc[0] == value_counts.max():
                        df.loc[df['outlier'] == labels[0], 'outlier'] = 'no'
                        df.loc[df['outlier'] == labels[1], 'outlier'] = 'yes'
                    else:
                        df.loc[df['outlier'] == labels[0], 'outlier'] = 'yes'
                        df.loc[df['outlier'] == labels[1], 'outlier'] = 'no'
            except Exception as e:
                print(f"Failed reading ARFF {dataset_path}: {e}")
                continue
        else:
            # Detect delimiter and read CSV
            try:
                delim = detect_delimiter(dataset_path)
                df = pd.read_csv(dataset_path, sep=delim)
            except Exception as e:
                print(f"Failed reading CSV {dataset_path}: {e}")
                continue

        # Prepare features and labels
        if 'outlier' not in df.columns:
            print(f"Dataset missing 'outlier' column, skipping: {dataset_path}")
            continue

        X = df.drop(columns=['outlier'])
        Y = df['outlier']
        df['dataset'] = dataset
        df['parameter'] = ''
        df['point'] = list(range(1, len(df) + 1))
        df['type'] = ['I' if t == 'no' else 'O' for t in Y]
        df['detect'] = ''
        num_outliers = int((Y == 'yes').sum())
        outlier_indices = Y[Y == 'yes'].index.tolist()
        inlier_indices = Y[Y == 'no'].index.tolist()

        # Iterate algorithms and compute scores/detections
        for a in algorithms:
            df_tmp = df.copy()
            algo_name = a['name']
            method = a['method']
            params = a['params']
            print(f'\t- {algo_name}')
            out_path = os.path.join('results', algo_name)

            # Skip if result file already exists
            if os.path.exists(os.path.join(out_path, dataset)):
                continue

            df_tmp['algorithm'] = algo_name

            # CompreX: expects categorical strings
            if method == CompreX:
                try:
                    instance = method()
                    X_cat = convert_numeric_to_string(X)
                    records = X_cat.to_dict(orient='records')
                    instance.transform(records)
                    instance.fit(records)
                    scores = [s['Score'] for s in instance.score(records)]
                    y_pred = top_n_indices(scores, n=num_outliers)
                    df_tmp['score'] = scores
                    df_tmp['ranking'] = numeric_ranking(scores, ascending=False)
                    df_tmp['detect'] = detection_outlier(Y, y_pred)
                except Exception as e:
                    print(f"CompreX failed for {dataset}: {e}")
                    continue

            # AVF: multiple bin parameter values
            elif method == AVF:
                df_list = []
                for p in params.get('bins', []):
                    try:
                        df_tmp2 = df_tmp.copy()
                        instance = method(X, bins=p)
                        scores = instance['Score'].tolist()
                        # For AVF lower score = outlier -> descending=False
                        y_pred = top_n_indices(scores, n=num_outliers, descending=False)
                        df_tmp2['score'] = scores
                        df_tmp2['ranking'] = numeric_ranking(scores, ascending=True)
                        df_tmp2['parameter'] = f'bins:{p}'
                        df_tmp2['detect'] = detection_outlier(Y, y_pred)
                        df_list.append(df_tmp2)
                    except Exception as e:
                        print(f"AVF failed with bins={p} on {dataset}: {e}")
                if df_list:
                    df_tmp = pd.concat(df_list, ignore_index=True)
                else:
                    continue

            # SCAN: algorithm instance with fit and obj_score
            elif method == SCAN:
                try:
                    instance = method()
                    for p in params:
                        df_tmp_local = df_tmp.copy()
                        df_tmp_local['parameter'] = ', '.join([f'{k}:{v}' for k, v in p.items()])
                        instance.fit(X.to_numpy(), **p)
                        scores = instance.obj_score
                        y_pred = top_n_indices(scores, n=num_outliers)
                        df_tmp_local['score'] = scores
                        df_tmp_local['ranking'] = numeric_ranking(scores, ascending=False)
                        df_tmp_local['detect'] = detection_outlier(Y, y_pred)
                        # Save per-parameter results immediately
                        save_result(df_tmp_local, out_path, dataset)
                    # Continue to next algorithm (results already saved)
                    continue
                except Exception as e:
                    print(f"SCAN failed on {dataset}: {e}")
                    continue

            # CBRW: coupled biased random walks
            elif method == CBRW:
                try:
                    instance = method()
                    records = X.to_dict(orient='records')
                    instance.add_observations(records)
                    instance.fit()
                    scores = instance.score(records)
                    y_pred = top_n_indices(scores, n=num_outliers)
                    df_tmp['score'] = scores
                    df_tmp['ranking'] = numeric_ranking(scores, ascending=False)
                    df_tmp['detect'] = detection_outlier(Y, y_pred)
                except Exception as e:
                    print(f"CBRW failed on {dataset}: {e}")
                    continue

            # FPOF: frequent pattern-based scoring
            elif method == FPOF:
                try:
                    df_list = []
                    for p in params:
                        df_tmp2 = df_tmp.copy()
                        df_tmp2['parameter'] = ', '.join([f'{k}:{v}' for k, v in p.items()])
                        scores = method(X, **p)['scores']
                        y_pred = top_n_indices(scores, n=num_outliers)
                        df_tmp2['score'] = scores
                        df_tmp2['ranking'] = numeric_ranking(scores, ascending=False)
                        df_tmp2['detect'] = detection_outlier(Y, y_pred)
                        df_list.append(df_tmp2)
                    if df_list:
                        df_tmp = pd.concat(df_list, ignore_index=True)
                    else:
                        continue
                except Exception as e:
                    print(f"FPOF failed on {dataset}: {e}")
                    continue

            # Compute basic correctness stats (kept for logging)
            try:
                correct_outliers = sorted(set(y_pred).intersection(set(outlier_indices)))
                false_outliers = sorted(set(y_pred).difference(set(outlier_indices)))
                correct_inliers = sorted(set(inlier_indices).difference(set(y_pred)))
                false_inliers = sorted(set(y_pred).intersection(set(inlier_indices)))
            except Exception:
                # If y_pred not defined, set empty lists
                correct_outliers = false_outliers = correct_inliers = false_inliers = []

            # Save computed results for this algorithm and dataset
            try:
                save_result(df_tmp, out_path, dataset)
            except Exception as e:
                print(f"Failed to save results for {algo_name} on {dataset}: {e}")

        # end algorithms loop
    # end datasets loop
# end repositories loop