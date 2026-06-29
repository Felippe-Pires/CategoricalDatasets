#!/usr/bin/env python3
"""
Runtime benchmark: CBRW, SDRW, IForest (PyOD) and KNN (PyOD).

Usage:
  python run_runtime_benchmark.py
  python run_runtime_benchmark.py --plot-only
  python run_runtime_benchmark.py --plot-only --scenario ad_nominal
  python run_runtime_benchmark.py --scenario covertype
  python run_runtime_benchmark.py --fresh          # ignore checkpoint and restart
  python run_runtime_benchmark.py --list-scenarios
  python run_runtime_benchmark.py --dataset covertype_test.csv --mode instances --class-column class

Dependencies (venv):
  pip install -r requirements.txt
  pip install coupled-biased-random-walks --no-deps   # CBRW (PyPI)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Configuration — edit here for new experiments
# -----------------------------------------------------------------------------

# Proportions for each collection step (10% … 100%)
PROPORTIONS: List[float] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

# Scenario 1 (instances): fraction of instances with class == outlier_class_value
OUTLIER_FRACTION: float = 0.05
ALLOW_OUTLIER_REPLACEMENT: bool = True

# Algorithm repetitions per collection step (runtime averaged), except KNN (see below)
N_RUNS_PER_SAMPLE: int = 10

# KNN: N_RUNS_PER_SAMPLE executions per n_neighbors value; mean = mean over all runs
KNN_N_NEIGHBORS_VALUES: List[int] = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

# Detection of aberrant timings before averaging (IQR method; MAD fallback)
TIMING_OUTLIER_IQR_MULTIPLIER: float = 1.5
TIMING_OUTLIER_MIN_RUNS: int = 4
TIMING_OUTLIER_MODIFIED_Z: float = 3.5

RANDOM_SEED: int = 42

ALGORITHMS: Tuple[str, ...] = ("CBRW", "SDRW", "IForest", "KNN")

# Optional row limit (None = full dataset; useful for quick tests)
MAX_ROWS: Optional[int] = None

OUTPUT_DIR_NAME: str = "../results/runtime_scalability"

# Extra scenarios in JSON (optional): databases/scenarios.json — list in the same format
SCENARIOS_JSON_FILE: str = "databases/scenarios.json"

# Built-in scenarios — mode: "instances" (sample rows) or "features" (sample columns)
SCENARIOS: List[Dict[str, Any]] = [
    {
        "name": "covertype",
        "dataset": "covertype.csv",
        "mode": "instances",
        "class_column": "class",
        "outlier_class_value": 1,
    },
    {
        "name": "ad_nominal",
        "dataset": "ad_nominal.csv",
        "mode": "features",
        "class_column": "outlier",
    },
    {
        "name": "w7a_libsvm",
        "dataset": "w7a_libsvm.csv",
        "mode": "features",
        "class_column": "class",
    },
]

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATABASES_DIR = ROOT / "databases"
SDRW_PYTHON = ROOT / "algorithm" / "SDRW_python"
OUTPUT_DIR = ROOT / OUTPUT_DIR_NAME


class ScenarioConfig(TypedDict, total=False):
    """Per-scenario settings: dataset path, sampling mode, and label column."""

    name: str
    dataset: str
    mode: str
    class_column: str
    outlier_class_value: Any


class ProportionTimingResult(TypedDict, total=False):
    """Timing stats for one algorithm at a single proportion step."""

    all_runs_seconds: List[float]
    excluded_runs_seconds: List[float]
    mean_seconds: float
    n_runs_total: int
    n_runs_used: int
    knn_n_neighbors_per_run: List[int]
    knn_runs_by_n_neighbors: Dict[str, List[float]]


ScenarioResults = Dict[str, Dict[float, ProportionTimingResult]]


def _setup_import_paths() -> None:
    """Add the local SDRW package directory to sys.path for imports."""
    sdrw_path = str(SDRW_PYTHON)
    if sdrw_path not in sys.path:
        sys.path.insert(0, sdrw_path)


def checkpoint_json_path(cfg: ScenarioConfig) -> Path:
    """Return the JSON checkpoint path for a scenario."""
    return OUTPUT_DIR / f"{cfg['name']}_runtime.json"


def config_fingerprint(
    cfg: ScenarioConfig,
    proportions_planned: List[float],
) -> dict:
    """Build a hashable snapshot of benchmark settings for checkpoint validation."""
    fp: dict = {
        "scenario": cfg["name"],
        "mode": cfg["mode"],
        "dataset": cfg["dataset"],
        "class_column": cfg["class_column"],
        "proportions_planned": proportions_planned,
        "n_runs_per_sample": N_RUNS_PER_SAMPLE,
        "random_seed": RANDOM_SEED,
        "max_rows": MAX_ROWS,
        "algorithms": list(ALGORITHMS),
        "knn_n_neighbors_values": KNN_N_NEIGHBORS_VALUES,
        "knn_runs_per_n_neighbors": N_RUNS_PER_SAMPLE,
        "timing_outlier_filter": {
            "iqr_multiplier": TIMING_OUTLIER_IQR_MULTIPLIER,
            "min_runs_to_filter": TIMING_OUTLIER_MIN_RUNS,
            "modified_z_threshold": TIMING_OUTLIER_MODIFIED_Z,
        },
    }
    if cfg["mode"] == "instances":
        fp["outlier_fraction"] = OUTLIER_FRACTION
        fp["outlier_class_value"] = cfg.get("outlier_class_value", 1)
        fp["allow_outlier_replacement"] = ALLOW_OUTLIER_REPLACEMENT
    return fp


def expected_n_runs(algorithm: str) -> int:
    """Return the number of timing runs required for a complete proportion step."""
    if algorithm == "KNN":
        return len(KNN_N_NEIGHBORS_VALUES) * N_RUNS_PER_SAMPLE
    return N_RUNS_PER_SAMPLE


def _deserialize_proportion_entry(entry: Any) -> ProportionTimingResult:
    """Restore a proportion timing entry from JSON (new or legacy scalar format)."""
    if isinstance(entry, dict) and "all_runs_seconds" in entry:
        result: ProportionTimingResult = {
            "all_runs_seconds": [float(x) for x in entry["all_runs_seconds"]],
            "excluded_runs_seconds": [
                float(x) for x in entry.get("excluded_runs_seconds", [])
            ],
            "mean_seconds": float(entry["mean_seconds"]),
            "n_runs_total": int(entry["n_runs_total"]),
            "n_runs_used": int(entry["n_runs_used"]),
        }
        if "knn_n_neighbors_per_run" in entry:
            result["knn_n_neighbors_per_run"] = [
                int(x) for x in entry["knn_n_neighbors_per_run"]
            ]
        elif "knn_n_neighbors" in entry:
            result["knn_n_neighbors_per_run"] = [int(x) for x in entry["knn_n_neighbors"]]
        if "knn_runs_by_n_neighbors" in entry:
            result["knn_runs_by_n_neighbors"] = {
                str(k): [float(v) for v in vals]
                for k, vals in entry["knn_runs_by_n_neighbors"].items()
            }
        return result
    val = float(entry)
    return {
        "all_runs_seconds": [val],
        "excluded_runs_seconds": [],
        "mean_seconds": val,
        "n_runs_total": 1,
        "n_runs_used": 1,
    }


def parse_scenario_results_from_payload(data: dict) -> ScenarioResults:
    """Parse timing results from a checkpoint or results JSON payload."""
    results: ScenarioResults = {alg: {} for alg in ALGORITHMS}
    source = data.get("times_by_proportion") or data.get("mean_times_seconds", {})
    for alg, props in source.items():
        if alg not in results:
            results[alg] = {}
        for p_str, entry in props.items():
            results[alg][float(p_str)] = _deserialize_proportion_entry(entry)
    return results


def is_proportion_complete(
    results: ScenarioResults,
    proportion: float,
) -> bool:
    """Return True when every algorithm has finished all runs for a proportion."""
    for alg in ALGORITHMS:
        timing = results.get(alg, {}).get(proportion)
        if timing is None:
            return False
        if timing["n_runs_total"] != expected_n_runs(alg):
            return False
        if alg == "KNN":
            by_n = timing.get("knn_runs_by_n_neighbors")
            if by_n:
                for n in KNN_N_NEIGHBORS_VALUES:
                    if len(by_n.get(str(n), [])) != N_RUNS_PER_SAMPLE:
                        return False
    return True


def completed_proportions(results: ScenarioResults) -> List[float]:
    """List proportion values that are fully complete across all algorithms."""
    props = set()
    for alg_times in results.values():
        props.update(alg_times.keys())
    return sorted(
        p for p in props if is_proportion_complete(results, p)
    )


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON via a temporary file to avoid partial writes on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def save_checkpoint(
    cfg: ScenarioConfig,
    results: ScenarioResults,
    proportions_planned: List[float],
    *,
    status: str = "in_progress",
) -> Path:
    """Persist partial or final results with metadata for resume support."""
    payload = results_to_payload(cfg, results)
    payload["status"] = status
    payload["config_fingerprint"] = config_fingerprint(cfg, proportions_planned)
    payload["completed_proportions"] = completed_proportions(results)
    path = checkpoint_json_path(cfg)
    write_json_atomic(path, payload)
    return path


def load_checkpoint(
    cfg: ScenarioConfig,
    proportions_planned: List[float],
    *,
    force_fresh: bool = False,
) -> ScenarioResults:
    """Load a saved checkpoint, or return empty results when starting fresh."""
    path = checkpoint_json_path(cfg)
    if force_fresh or not path.is_file():
        return {alg: {} for alg in ALGORITHMS}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"WARNING: corrupted checkpoint ({path.name}): {exc}. Restarting.")
        return {alg: {} for alg in ALGORITHMS}

    if data.get("scenario") != cfg["name"]:
        return {alg: {} for alg in ALGORITHMS}

    stored_fp = data.get("config_fingerprint")
    current_fp = config_fingerprint(cfg, proportions_planned)
    if stored_fp is not None and stored_fp != current_fp:
        print(
            f"\nWARNING: configuration changed since the last checkpoint for {cfg['name']}. "
            "Previous collections will be ignored. Use --fresh to suppress this warning."
        )
        return {alg: {} for alg in ALGORITHMS}

    results = parse_scenario_results_from_payload(data)
    done = completed_proportions(results)
    if done:
        print(
            f"\nCheckpoint: {len(done)} collection(s) already completed in {path.name} "
            f"({', '.join(f'{p:.0%}' for p in done)})"
        )
    return results


def scenarios_json_path() -> Path:
    """Return the path to the optional extra scenarios JSON file."""
    return DATABASES_DIR / SCENARIOS_JSON_FILE


def load_scenario_definitions() -> List[Dict[str, Any]]:
    """Built-in scenarios (SCENARIOS) + optional databases/scenarios.json."""
    definitions = list(SCENARIOS)
    json_path = scenarios_json_path()
    if not json_path.is_file():
        return definitions

    extra = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(extra, list):
        raise ValueError(f"{json_path}: content must be a JSON list of scenarios.")

    names_seen = {d["name"] for d in definitions}
    for item in extra:
        if not isinstance(item, dict) or "name" not in item or "dataset" not in item:
            raise ValueError(f"{json_path}: each scenario must have 'name' and 'dataset'.")
        if item["name"] in names_seen:
            raise ValueError(
                f"Duplicate scenario '{item['name']}' in {json_path.name} "
                f"(already defined in SCENARIOS or a previous JSON entry)."
            )
        names_seen.add(item["name"])
        definitions.append(item)
    return definitions


def list_available_scenarios() -> List[ScenarioConfig]:
    """Return all built-in and JSON-defined scenarios, fully resolved."""
    return [resolve_scenario(raw) for raw in load_scenario_definitions()]


def resolve_scenario(raw: Dict[str, Any]) -> ScenarioConfig:
    """Normalize a raw scenario dict into a ScenarioConfig."""
    mode = raw.get("mode", "instances")
    if mode not in ("instances", "features"):
        raise ValueError(
            f"Scenario {raw.get('name')}: mode must be 'instances' or 'features'."
        )
    cfg: ScenarioConfig = {
        "name": raw["name"],
        "dataset": raw["dataset"],
        "mode": mode,
        "class_column": raw.get("class_column", "class"),
    }
    if mode == "instances":
        cfg["outlier_class_value"] = raw.get("outlier_class_value", 1)
    return cfg


def resolve_dataset_path(dataset: str) -> Path:
    """
    Resolve a CSV in databases/, as a project-relative path, or as an absolute path.
    """
    raw = Path(dataset)
    if raw.is_absolute() and raw.is_file():
        return raw.resolve()

    if raw.is_file():
        return raw.resolve()

    in_databases = (DATABASES_DIR / raw.name).resolve()
    if in_databases.is_file():
        return in_databases

    from_root = (ROOT / raw).resolve()
    if from_root.is_file():
        return from_root

    raise FileNotFoundError(
        f"Dataset not found: {dataset!r}. "
        f"Place the CSV in {DATABASES_DIR} or provide the full path."
    )


def dataset_label_for_storage(dataset_path: Path) -> str:
    """Identifier stored in the checkpoint (relative to databases/ when possible)."""
    try:
        rel = dataset_path.resolve().relative_to(DATABASES_DIR.resolve())
        return rel.as_posix()
    except ValueError:
        return str(dataset_path.resolve())


def scenario_from_cli_args(args: argparse.Namespace) -> ScenarioConfig:
    """Build a one-off scenario from --dataset and related CLI flags."""
    if not args.dataset:
        raise ValueError("Provide --dataset to create a scenario from the command line.")

    path = resolve_dataset_path(args.dataset)
    mode = args.mode
    if mode not in ("instances", "features"):
        raise ValueError("--mode must be 'instances' or 'features'.")

    name = args.name or path.stem
    raw: Dict[str, Any] = {
        "name": name,
        "dataset": dataset_label_for_storage(path),
        "mode": mode,
        "class_column": args.class_column,
    }
    if mode == "instances":
        raw["outlier_class_value"] = args.outlier_class_value
    return resolve_scenario(raw)


def load_full_dataset(csv_path: Path) -> pd.DataFrame:
    """Load a CSV dataset, optionally truncated by MAX_ROWS."""
    df = pd.read_csv(csv_path, low_memory=False)
    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()
    return df


def feature_column_names(df: pd.DataFrame, class_column: str) -> List[str]:
    """Return all columns except the class/outlier label column."""
    return [c for c in df.columns if c != class_column]


def split_pools(
    df: pd.DataFrame,
    class_column: str,
    outlier_class_value: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split row indices into outlier and inlier pools based on class value."""
    outlier_mask = df[class_column] == outlier_class_value
    outlier_idx = np.flatnonzero(outlier_mask.to_numpy())
    inlier_idx = np.flatnonzero((~outlier_mask).to_numpy())
    return outlier_idx, inlier_idx


def max_feasible_proportion(
    n_rows: int,
    n_outliers: int,
    n_inliers: int,
) -> float:
    """Upper bound on sampling proportion given outlier/inlier pool sizes."""
    if n_rows == 0:
        return 0.0
    if ALLOW_OUTLIER_REPLACEMENT:
        cap_out = 1.0
    else:
        # Without replacement, outlier pool size limits the max proportion
        cap_out = n_outliers / (OUTLIER_FRACTION * n_rows) if OUTLIER_FRACTION > 0 else 1.0
    inlier_fraction = 1.0 - OUTLIER_FRACTION
    cap_in = (
        n_inliers / (inlier_fraction * n_rows) if inlier_fraction > 0 else 1.0
    )
    return float(min(1.0, cap_out, cap_in))


def filter_proportions_instances(
    n_rows: int,
    n_outliers: int,
    n_inliers: int,
) -> List[float]:
    """Keep only proportions feasible for instance-mode sampling."""
    max_p = max_feasible_proportion(n_rows, n_outliers, n_inliers)
    feasible = [p for p in PROPORTIONS if p <= max_p + 1e-12]
    skipped = [p for p in PROPORTIONS if p not in feasible]
    if skipped:
        hint = (
            " Set ALLOW_OUTLIER_REPLACEMENT=True to keep all proportions."
            if not ALLOW_OUTLIER_REPLACEMENT
            else ""
        )
        print(
            f"\nWARNING: {len(skipped)} proportion(s) skipped — "
            f"with {OUTLIER_FRACTION:.0%} outliers only ~{max_p:.1%} is feasible "
            f"({n_outliers:,} unique outliers in {n_rows:,} rows).{hint}"
        )
        print(f"  Skipped: {[f'{p:.0%}' for p in skipped]}")
    return feasible


def build_instance_sample(
    df: pd.DataFrame,
    class_column: str,
    outlier_class_value: Any,
    outlier_idx: np.ndarray,
    inlier_idx: np.ndarray,
    proportion: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sample rows to reach the target proportion while preserving OUTLIER_FRACTION.

    Outliers may be drawn with replacement when ALLOW_OUTLIER_REPLACEMENT is True.
    """
    n_total = max(1, int(np.floor(proportion * len(df))))
    n_outliers = int(round(OUTLIER_FRACTION * n_total))
    n_outliers = min(max(n_outliers, 0), n_total)
    n_inliers = n_total - n_outliers

    if n_outliers > len(outlier_idx):
        if not ALLOW_OUTLIER_REPLACEMENT:
            raise ValueError(
                f"Proportion {proportion:.0%}: need {n_outliers} outliers "
                f"({class_column}={outlier_class_value}), "
                f"but the dataset only has {len(outlier_idx)}."
            )
        replace_out = True
    else:
        replace_out = False
    if n_inliers > len(inlier_idx):
        raise ValueError(
            f"Proportion {proportion:.0%}: need {n_inliers} inliers, "
            f"but the dataset only has {len(inlier_idx)}."
        )

    chosen_out = rng.choice(outlier_idx, size=n_outliers, replace=replace_out)
    chosen_in = (
        rng.choice(inlier_idx, size=n_inliers, replace=False)
        if n_inliers
        else np.array([], dtype=int)
    )
    indices = np.concatenate([chosen_out, chosen_in])
    rng.shuffle(indices)
    return df.iloc[indices].reset_index(drop=True)


def build_feature_sample(
    df: pd.DataFrame,
    class_column: str,
    proportion: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Select a random fraction of the feature columns; class_column is always included."""
    features = feature_column_names(df, class_column)
    n_pick = max(1, int(np.floor(proportion * len(features))))
    n_pick = min(n_pick, len(features))
    chosen = rng.choice(features, size=n_pick, replace=False)
    cols = list(chosen) + [class_column]
    return df[cols].copy()


def dataframe_to_observations(
    df: pd.DataFrame,
    class_column: str,
) -> List[Dict[str, str]]:
    """Convert feature rows to CBRW observation dicts (string values)."""
    feature_cols = feature_column_names(df, class_column)
    observations: List[Dict[str, str]] = []
    for row in df[feature_cols].itertuples(index=False, name=None):
        observations.append({col: str(val) for col, val in zip(feature_cols, row)})
    return observations


def _as_nominal_series(series: pd.Series) -> Tuple[List[str], np.ndarray]:
    """Encode a categorical column as integer indices."""
    uniq = sorted(series.dropna().astype(str).unique().tolist())
    index_map = {v: i for i, v in enumerate(uniq)}
    encoded = series.astype(str).map(index_map).astype(float).to_numpy()
    return uniq, encoded


def _is_nominal_column(series: pd.Series) -> bool:
    """Heuristic: object/category dtypes or low cardinality are treated as nominal."""
    if series.dtype == object or str(series.dtype).startswith("category"):
        return True
    return series.nunique(dropna=True) <= 20


def dataframe_to_sdrw_dataset(df: pd.DataFrame):
    """Build an SDRW Dataset with nominal attributes auto-detected per column."""
    from sdrw.dataset import Attribute, Dataset

    attributes: List = []
    columns: List[np.ndarray] = []

    for col_name in df.columns:
        series = df[col_name]
        if _is_nominal_column(series):
            uniq, encoded = _as_nominal_series(series)
            attributes.append(Attribute(str(col_name), uniq))
            columns.append(encoded)
        else:
            attributes.append(Attribute(str(col_name), []))
            columns.append(series.astype(float).to_numpy())

    data = np.column_stack(columns)
    dataset = Dataset(attributes, data)
    dataset.set_class_index(dataset.num_attributes() - 1)
    return dataset


def feature_matrix(df: pd.DataFrame, class_column: str) -> np.ndarray:
    """Return a float64 feature matrix for PyOD algorithms (IForest, KNN)."""
    cols = feature_column_names(df, class_column)
    return df[cols].astype(np.float64).to_numpy()


def robust_mean_runtime(durations: List[float]) -> Tuple[float, List[float], List[float]]:
    """
    Compute the mean excluding aberrant timings (measurement outliers).
    Returns (mean, timings used, timings excluded).
    """
    arr = np.asarray(durations, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, [], []

    if n < TIMING_OUTLIER_MIN_RUNS:
        return float(np.mean(arr)), arr.tolist(), []

    q1, q3 = np.percentile(arr, [25, 75])
    iqr = float(q3 - q1)

    if iqr > 0:
        # Primary filter: interquartile range (Tukey fences)
        low = q1 - TIMING_OUTLIER_IQR_MULTIPLIER * iqr
        high = q3 + TIMING_OUTLIER_IQR_MULTIPLIER * iqr
        mask = (arr >= low) & (arr <= high)
    else:
        # Fallback when IQR is zero: modified z-score based on MAD
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        if mad == 0:
            return float(np.mean(arr)), arr.tolist(), []
        modified_z = 0.6745 * (arr - median) / mad
        mask = np.abs(modified_z) <= TIMING_OUTLIER_MODIFIED_Z

    kept = arr[mask]
    excluded = arr[~mask]

    if len(kept) == 0:
        return float(np.mean(arr)), arr.tolist(), []

    return float(np.mean(kept)), kept.tolist(), excluded.tolist()


def make_timers(class_column: str) -> Dict[str, Callable[[pd.DataFrame], float]]:
    """Create per-algorithm timing callables (fit + score pipeline)."""

    def time_cbrw(df: pd.DataFrame) -> float:
        """Time CBRW fit + score on nominal observation dicts."""
        from coupled_biased_random_walks import CBRW

        observations = dataframe_to_observations(df, class_column)
        detector = CBRW()
        t0 = time.perf_counter()
        detector.add_observations(observations)
        detector.fit()
        detector.score(observations)
        return time.perf_counter() - t0

    def time_sdrw(df: pd.DataFrame) -> float:
        """Time the full SDRW pipeline (centroids + outlierness learning + scoring)."""
        from sdrw.dsvl import DSVL
        from sdrw.value_centroid import ValueCentroid

        dataset = dataframe_to_sdrw_dataset(df)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            vc = ValueCentroid()
            cp_list = vc.initial_centroid_list(dataset)
            cp_list = vc.generate_coupled_centroids(cp_list, dataset)
            vc.obtain_global_centroid(cp_list, dataset)
            dsvl = DSVL(cp_list)
            dsvl.value_outlierness_learning(dataset.num_instances)
            dsvl.scoring_test_instances(dataset)
        return time.perf_counter() - t0

    def time_iforest(df: pd.DataFrame) -> float:
        """Time PyOD Isolation Forest fit on the numeric feature matrix."""
        from pyod.models.iforest import IForest

        x = feature_matrix(df, class_column)
        model = IForest()
        t0 = time.perf_counter()
        model.fit(x)
        _ = model.decision_scores_
        return time.perf_counter() - t0

    return {
        "CBRW": time_cbrw,
        "SDRW": time_sdrw,
        "IForest": time_iforest,
    }


def time_knn_single_run(df: pd.DataFrame, class_column: str, n_neighbors: int) -> float:
    """Time one KNN fit on the sample for a given n_neighbors value."""
    from pyod.models.knn import KNN

    x = feature_matrix(df, class_column)
    model = KNN(n_neighbors=n_neighbors)
    t0 = time.perf_counter()
    model.fit(x)
    _ = model.decision_scores_
    return time.perf_counter() - t0


def run_knn_on_sample(
    sample: pd.DataFrame,
    class_column: str,
) -> ProportionTimingResult:
    """N_RUNS_PER_SAMPLE executions per n_neighbors; mean = mean over all runs."""
    durations: List[float] = []
    n_per_run: List[int] = []
    runs_by_n: Dict[str, List[float]] = {}

    for n_neighbors in KNN_N_NEIGHBORS_VALUES:
        runs: List[float] = []
        for run in range(N_RUNS_PER_SAMPLE):
            elapsed = time_knn_single_run(sample, class_column, n_neighbors)
            runs.append(elapsed)
            durations.append(elapsed)
            n_per_run.append(n_neighbors)
            print(
                f"  KNN n_neighbors={n_neighbors} "
                f"run {run + 1}/{N_RUNS_PER_SAMPLE}: {elapsed:.3f}s"
            )
        runs_by_n[str(n_neighbors)] = runs

    mean_time = float(np.mean(durations))
    n_total = len(KNN_N_NEIGHBORS_VALUES) * N_RUNS_PER_SAMPLE
    result: ProportionTimingResult = {
        "all_runs_seconds": durations,
        "excluded_runs_seconds": [],
        "mean_seconds": mean_time,
        "n_runs_total": n_total,
        "n_runs_used": n_total,
        "knn_n_neighbors_per_run": n_per_run,
        "knn_runs_by_n_neighbors": runs_by_n,
    }
    print(
        f"  -> KNN mean: {mean_time:.3f}s "
        f"(arithmetic mean over {len(durations)} runs: "
        f"{len(KNN_N_NEIGHBORS_VALUES)} x {N_RUNS_PER_SAMPLE} n_neighbors)"
    )
    return result


def run_algorithms_on_sample(
    sample: pd.DataFrame,
    proportion: float,
    prop_idx: int,
    n_proportions: int,
    timers: Dict[str, Callable[[pd.DataFrame], float]],
    collection_label: str,
    class_column: str,
) -> Dict[str, ProportionTimingResult]:
    """Run all algorithms on one sampled subset and aggregate timing stats."""
    results: Dict[str, ProportionTimingResult] = {}
    print(f"\nCollection {prop_idx + 1}/{n_proportions} — {collection_label}")

    for alg_name in ALGORITHMS:
        if alg_name == "KNN":
            results["KNN"] = run_knn_on_sample(sample, class_column)
            continue
        if alg_name not in timers:
            continue
        timer = timers[alg_name]
        durations: List[float] = []
        for run in range(N_RUNS_PER_SAMPLE):
            elapsed = timer(sample)
            durations.append(elapsed)
            print(f"  {alg_name} run {run + 1}/{N_RUNS_PER_SAMPLE}: {elapsed:.3f}s")

        mean_time, kept, excluded = robust_mean_runtime(durations)
        results[alg_name] = {
            "all_runs_seconds": durations,
            "excluded_runs_seconds": excluded,
            "mean_seconds": mean_time,
            "n_runs_total": len(durations),
            "n_runs_used": len(kept),
        }
        if excluded:
            print(
                f"  -> {alg_name} mean: {mean_time:.3f}s "
                f"({len(kept)}/{len(durations)} runs; "
                f"excluded: {[f'{t:.3f}' for t in excluded]})"
            )
        else:
            print(f"  -> {alg_name} mean: {mean_time:.3f}s ({len(kept)} runs)")

    return results


def _merge_and_checkpoint(
    cfg: ScenarioConfig,
    all_results: ScenarioResults,
    proportion: float,
    prop_results: Dict[str, ProportionTimingResult],
    proportions_planned: List[float],
) -> Path:
    """Merge one proportion's results and write an incremental checkpoint."""
    for alg, timing in prop_results.items():
        all_results[alg][proportion] = timing
    path = save_checkpoint(
        cfg, all_results, proportions_planned, status="in_progress"
    )
    print(f"  Checkpoint saved: {path}")
    return path


def run_scenario_instances(
    cfg: ScenarioConfig,
    df: pd.DataFrame,
    timers: Dict[str, Callable[[pd.DataFrame], float]],
    *,
    force_fresh: bool = False,
) -> Tuple[ScenarioResults, List[float]]:
    """Benchmark instance-mode sampling: vary the number of rows per proportion."""
    class_col = cfg["class_column"]
    outlier_val = cfg.get("outlier_class_value", 1)

    outlier_idx, inlier_idx = split_pools(df, class_col, outlier_val)
    print(
        f"Mode: instances | {len(df):,} rows | "
        f"outliers ({class_col}={outlier_val}): {len(outlier_idx):,} | "
        f"inliers: {len(inlier_idx):,}"
    )

    proportions = filter_proportions_instances(len(df), len(outlier_idx), len(inlier_idx))
    if not proportions:
        raise ValueError(
            f"No feasible proportion for {cfg['dataset']}. "
            f"Adjust OUTLIER_FRACTION, PROPORTIONS, or ALLOW_OUTLIER_REPLACEMENT."
        )

    max_prop = max(PROPORTIONS)
    need_out = int(
        round(OUTLIER_FRACTION * max(1, int(np.floor(max_prop * len(df)))))
    )
    if ALLOW_OUTLIER_REPLACEMENT and need_out > len(outlier_idx):
        print(
            f"\nNote: high proportions will sample outliers with replacement "
            f"({len(outlier_idx):,} unique; up to {need_out:,} at {max_prop:.0%})."
        )

    all_results = load_checkpoint(cfg, proportions, force_fresh=force_fresh)
    done_set = set(completed_proportions(all_results))

    for prop_idx, proportion in enumerate(proportions):
        if proportion in done_set:
            print(
                f"\nCollection {prop_idx + 1}/{len(proportions)} — "
                f"{proportion:.0%} (already completed, skipping)"
            )
            continue

        rng = np.random.default_rng(RANDOM_SEED + prop_idx)
        sample = build_instance_sample(
            df, class_col, outlier_val, outlier_idx, inlier_idx, proportion, rng
        )
        rate = (sample[class_col] == outlier_val).mean()
        label = (
            f"{proportion:.0%} ({len(sample):,} instances, outliers {rate:.2%})"
        )
        prop_results = run_algorithms_on_sample(
            sample, proportion, prop_idx, len(proportions), timers, label, class_col
        )
        _merge_and_checkpoint(
            cfg, all_results, proportion, prop_results, proportions
        )

    return all_results, proportions


def run_scenario_features(
    cfg: ScenarioConfig,
    df: pd.DataFrame,
    timers: Dict[str, Callable[[pd.DataFrame], float]],
    *,
    force_fresh: bool = False,
) -> Tuple[ScenarioResults, List[float]]:
    """Benchmark feature-mode sampling: vary the number of feature columns."""
    class_col = cfg["class_column"]
    features = feature_column_names(df, class_col)
    n_rows = len(df)

    print(
        f"Mode: features | {n_rows:,} instances (all rows) | "
        f"{len(features):,} features | target column always included: {class_col!r}"
    )

    proportions = list(PROPORTIONS)
    all_results = load_checkpoint(cfg, proportions, force_fresh=force_fresh)
    done_set = set(completed_proportions(all_results))

    for prop_idx, proportion in enumerate(proportions):
        if proportion in done_set:
            print(
                f"\nCollection {prop_idx + 1}/{len(proportions)} — "
                f"{proportion:.0%} (already completed, skipping)"
            )
            continue

        rng = np.random.default_rng(RANDOM_SEED + prop_idx)
        sample = build_feature_sample(df, class_col, proportion, rng)
        n_feat = len(feature_column_names(sample, class_col))
        label = (
            f"{proportion:.0%} ({n_feat:,} of {len(features):,} features + {class_col})"
        )
        prop_results = run_algorithms_on_sample(
            sample, proportion, prop_idx, len(proportions), timers, label, class_col
        )
        _merge_and_checkpoint(
            cfg, all_results, proportion, prop_results, proportions
        )

    return all_results, proportions


def run_scenario(
    cfg: ScenarioConfig,
    timers: Dict[str, Callable[[pd.DataFrame], float]],
    *,
    force_fresh: bool = False,
) -> Tuple[ScenarioResults, List[float]]:
    """Run a full scenario in either instances or features mode."""
    csv_path = resolve_dataset_path(cfg["dataset"])

    print(f"\n=== Scenario: {cfg['name']} ({csv_path.name}) ===")
    if force_fresh:
        ckpt = checkpoint_json_path(cfg)
        if ckpt.is_file():
            print(f"--fresh: ignoring existing checkpoint ({ckpt.name})")

    df = load_full_dataset(csv_path)

    if cfg["mode"] == "instances":
        return run_scenario_instances(cfg, df, timers, force_fresh=force_fresh)
    return run_scenario_features(cfg, df, timers, force_fresh=force_fresh)


def _serialize_proportion_entry(entry: Union[float, ProportionTimingResult]) -> dict:
    """Convert a proportion timing result to a JSON-serializable dict."""
    if isinstance(entry, dict):
        out = {
            "all_runs_seconds": entry["all_runs_seconds"],
            "excluded_runs_seconds": entry["excluded_runs_seconds"],
            "mean_seconds": entry["mean_seconds"],
            "n_runs_total": entry["n_runs_total"],
            "n_runs_used": entry["n_runs_used"],
        }
        if "knn_n_neighbors_per_run" in entry:
            out["knn_n_neighbors_per_run"] = entry["knn_n_neighbors_per_run"]
        if "knn_runs_by_n_neighbors" in entry:
            out["knn_runs_by_n_neighbors"] = entry["knn_runs_by_n_neighbors"]
        return out
    return {
        "all_runs_seconds": [float(entry)],
        "excluded_runs_seconds": [],
        "mean_seconds": float(entry),
        "n_runs_total": 1,
        "n_runs_used": 1,
    }


def mean_times_for_plot(results: ScenarioResults) -> Dict[str, Dict[float, float]]:
    """Extract mean runtimes per algorithm and proportion for plotting."""
    return {
        alg: {p: timing["mean_seconds"] for p, timing in prop_times.items()}
        for alg, prop_times in results.items()
    }


def results_to_payload(cfg: ScenarioConfig, results: ScenarioResults) -> dict:
    """Build the full JSON payload written to checkpoint/result files."""
    by_proportion = {
        alg: {str(p): _serialize_proportion_entry(timing) for p, timing in prop_times.items()}
        for alg, prop_times in results.items()
    }
    payload: dict = {
        "scenario": cfg["name"],
        "mode": cfg["mode"],
        "dataset": cfg["dataset"],
        "class_column": cfg["class_column"],
        "proportions": PROPORTIONS,
        "n_runs_per_sample": N_RUNS_PER_SAMPLE,
        "knn_n_neighbors_values": KNN_N_NEIGHBORS_VALUES,
        "knn_runs_per_n_neighbors": N_RUNS_PER_SAMPLE,
        "timing_outlier_filter": {
            "method": "IQR (fallback MAD modified z-score)",
            "iqr_multiplier": TIMING_OUTLIER_IQR_MULTIPLIER,
            "min_runs_to_filter": TIMING_OUTLIER_MIN_RUNS,
            "modified_z_threshold": TIMING_OUTLIER_MODIFIED_Z,
        },
        "times_by_proportion": by_proportion,
        "mean_times_seconds": {
            alg: {str(p): entry["mean_seconds"] for p, entry in props.items()}
            for alg, props in by_proportion.items()
        },
    }
    if cfg["mode"] == "instances":
        payload["outlier_fraction"] = OUTLIER_FRACTION
        payload["outlier_class_value"] = cfg.get("outlier_class_value", 1)
        payload["allow_outlier_replacement"] = ALLOW_OUTLIER_REPLACEMENT
    return payload


def load_results_from_json(
    json_path: Path,
) -> Tuple[str, str, Dict[str, Dict[float, float]]]:
    """Load mean runtimes for plotting (compatible with old and new JSON formats)."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    scenario = data["scenario"]
    mode = data.get("mode", "instances")

    if "times_by_proportion" in data:
        results: Dict[str, Dict[float, float]] = {}
        for alg, props in data["times_by_proportion"].items():
            results[alg] = {}
            for p_str, entry in props.items():
                if isinstance(entry, dict):
                    results[alg][float(p_str)] = float(entry["mean_seconds"])
                else:
                    results[alg][float(p_str)] = float(entry)
        return scenario, mode, results

    results = {}
    for alg, prop_times in data["mean_times_seconds"].items():
        results[alg] = {float(k): float(v) for k, v in prop_times.items()}
    return scenario, mode, results


def plot_results(
    results: Dict[str, Dict[float, float]],
    scenario_name: str,
    plot_path: Path,
    mode: str = "instances",
) -> None:
    """Generate an interactive Plotly line chart of runtime vs. dataset proportion."""
    proportions = sorted({p for prop_times in results.values() for p in prop_times})
    x_pct = [p * 100 for p in proportions]

    x_title = (
        "Dataset Dimensionality (%)"
        if mode == "features"
        else "Dataset Size (%)"
    )

    fig = go.Figure()
    for alg_name in sorted(results.keys()):
        prop_times = results[alg_name]
        y = [prop_times[p] for p in proportions]
        fig.add_trace(
            go.Scatter(
                x=x_pct,
                y=y,
                mode="lines+markers",
                name=alg_name,
                hovertemplate=(
                    "Proportion: %{x:,.0f}%<br>"
                    "Avg. runtime: %{y:,.3f} s<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Dataset {scenario_name}",
            x=0.5,
            xanchor="center",
            font=dict(size=24),
        ),
        xaxis_title=x_title,
        yaxis_title="Average Runtime (s)",
        template="plotly_white",
        legend_title_text="Algorithm",
        hovermode="x unified",
        width=1000,
        height=600,
        font=dict(size=16),
        legend=dict(font=dict(size=16), title_font=dict(size=17)),
    )
    fig.update_xaxes(
        ticksuffix="%",
        tickformat=",",
        tickfont=dict(size=16),
        title_font=dict(size=19),
    )
    fig.update_yaxes(
        tickformat=",",
        tickfont=dict(size=16),
        title_font=dict(size=19),
    )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = plot_path.suffix.lower()
    if suffix == ".html":
        fig.write_html(str(plot_path), include_plotlyjs="cdn")
    elif suffix in (".png", ".pdf", ".svg"):
        fig.write_image(str(plot_path))
    else:
        html_path = plot_path.with_suffix(".html")
        fig.write_html(str(html_path), include_plotlyjs="cdn")


def save_results(
    cfg: ScenarioConfig,
    results: ScenarioResults,
    proportions_planned: List[float],
) -> Tuple[Path, Path]:
    """Mark the scenario complete, save JSON, and write the Plotly HTML chart."""
    json_path = save_checkpoint(
        cfg, results, proportions_planned, status="completed"
    )

    plot_path = checkpoint_json_path(cfg).with_suffix(".html")
    plot_results(mean_times_for_plot(results), cfg["name"], plot_path, cfg["mode"])
    return json_path, plot_path


def discover_result_json_files(
    results_dir: Path,
    scenario: Optional[str] = None,
) -> List[Path]:
    """Find *_runtime.json result files, optionally filtered by scenario name."""
    if scenario:
        path = results_dir / f"{scenario}_runtime.json"
        if not path.is_file():
            raise FileNotFoundError(f"Results file not found: {path}")
        return [path]
    files = sorted(results_dir.glob("*_runtime.json"))
    if not files:
        raise FileNotFoundError(
            f"No *_runtime.json files in {results_dir}. Run the benchmark first."
        )
    return files


def run_plot_only(results_dir: Path, scenario: Optional[str] = None) -> None:
    """Regenerate Plotly HTML charts from existing JSON result files."""
    json_files = discover_result_json_files(results_dir, scenario)
    print(f"Generating {len(json_files)} Plotly chart(s) in {results_dir}")
    for json_path in json_files:
        scenario_name, mode, results = load_results_from_json(json_path)
        html_path = results_dir / f"{scenario_name}_runtime.html"
        plot_results(results, scenario_name, html_path, mode)
        print(f"  {html_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runtime benchmark (CBRW, SDRW, IForest, KNN).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Generate Plotly charts only from existing JSON files in the results dir.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=f"Directory for result JSON files (default: {OUTPUT_DIR_NAME}).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run or plot only this scenario (e.g. covertype, ad_nominal).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore partial checkpoint and rerun all collections from scratch.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List configured scenarios (SCENARIOS + scenarios.json) and exit.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="CSV",
        help="Dataset: file in databases/, relative path, or absolute path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["instances", "features"],
        default=None,
        help="Sampling mode (required with --dataset).",
    )
    parser.add_argument(
        "--class-column",
        type=str,
        default="class",
        help="Label/outlier column (with --dataset; default: class).",
    )
    parser.add_argument(
        "--outlier-class-value",
        type=int,
        default=1,
        help="Outlier class value in instances mode (default: 1).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Scenario name when using --dataset (default: CSV file stem).",
    )
    return parser.parse_args()


def select_scenarios(
    name_filter: Optional[str],
    *,
    cli_scenario: Optional[ScenarioConfig] = None,
) -> List[ScenarioConfig]:
    """Resolve which scenarios to run from CLI flags and optional name filter."""
    if cli_scenario is not None:
        if name_filter is not None and cli_scenario["name"] != name_filter:
            raise ValueError(
                f"--scenario {name_filter!r} does not match the CLI scenario "
                f"'{cli_scenario['name']}' (--name / --dataset)."
            )
        return [cli_scenario]

    configs = list_available_scenarios()
    if name_filter is None:
        return configs
    matched = [c for c in configs if c["name"] == name_filter]
    if not matched:
        names = ", ".join(c["name"] for c in configs)
        raise ValueError(f"Scenario '{name_filter}' not found. Available: {names}")
    return matched


def print_scenario_catalog() -> None:
    """Print all configured scenarios and whether their dataset file exists."""
    print("Available scenarios:\n")
    for cfg in list_available_scenarios():
        path = resolve_dataset_path(cfg["dataset"])
        exists = "ok" if path.is_file() else "FILE MISSING"
        extra = ""
        if cfg["mode"] == "instances":
            extra = (
                f", outlier={cfg.get('outlier_class_value', 1)} "
                f"in column {cfg['class_column']!r}"
            )
        print(
            f"  {cfg['name']}: {cfg['dataset']} [{cfg['mode']}{extra}] ({exists})"
        )
    json_path = scenarios_json_path()
    print(f"\nAdd scenarios in SCENARIOS (script) or in {json_path}")
    print("Or use: --dataset <file.csv> --mode instances|features [--class-column COL]")


def main() -> None:
    """CLI entry point: list scenarios, plot-only, or run benchmarks."""
    args = parse_args()
    results_dir = args.results_dir or OUTPUT_DIR

    if args.list_scenarios:
        print_scenario_catalog()
        return

    if args.plot_only:
        run_plot_only(results_dir, args.scenario)
        return

    if args.dataset and not args.mode:
        raise SystemExit("Error: use --mode instances or --mode features with --dataset.")

    cli_scenario = scenario_from_cli_args(args) if args.dataset else None

    _setup_import_paths()

    for cfg in select_scenarios(args.scenario, cli_scenario=cli_scenario):
        timers = make_timers(cfg["class_column"])
        results, proportions_planned = run_scenario(
            cfg, timers, force_fresh=args.fresh
        )
        pending = [
            p
            for p in proportions_planned
            if not is_proportion_complete(results, p)
        ]
        if pending:
            print(
                f"\nScenario {cfg['name']} incomplete "
                f"({len(pending)} collection(s) pending). "
                f"Run again to resume."
            )
            print(f"  Checkpoint: {checkpoint_json_path(cfg)}")
            continue

        json_path, plot_path = save_results(cfg, results, proportions_planned)
        print(f"\nResults saved:\n  {json_path}\n  {plot_path}")


if __name__ == "__main__":
    main()
