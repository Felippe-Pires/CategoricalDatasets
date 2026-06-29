"""
Main entry point for the SDRW (DSVL) outlier detection algorithm.
Python port of edu.uts.aai.utils.DSVL4ODUtils.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional

from .csv_utils import CSV
from .dataset import Dataset, divide, load_dataset
from .dsvl import DSVL
from .value_centroid import ValueCentroid

# Defaults mirror the Java project; override via CLI or edit here.
DIR_PATH = os.environ.get(
    "SDRW_DATA_DIR",
    r"databases",
)
DIR_RESULT = os.environ.get(
    "SDRW_RESULT_DIR",
    r"results",
)

train_instances: Optional[Dataset] = None
test_instances: Optional[Dataset] = None
data_set_full_name_list: List[str] = []
data_set_name_list: List[str] = []
data_filename = ""
data_set_name = ""
dir_executing = ""
result_list: Optional[List[List[str]]] = None
auc_score = 0.0
runtime_ms = 0
fs = False


def main(argv: Optional[List[str]] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    global fs, DIR_PATH

    if len(argv) >= 1:
        DIR_PATH = argv[0]
    if len(argv) >= 2:
        fs = argv[1].lower() in ("true", "1", "yes")
    if len(argv) >= 3 and not os.path.isdir(DIR_PATH):
        pass  # third arg was dirPath in Java when using args; paths handled below

    paths = [p.strip() for p in DIR_PATH.split(",") if p.strip()]
    for path in paths:
        global dir_executing
        dir_executing = path
        data_file_is_dir = os.path.isdir(path)
        if data_file_is_dir:
            build_data_sets_path_list(path)
        else:
            global data_filename
            data_filename = path
        value_selection_options(data_file_is_dir)


def value_selection_options(flag: bool) -> None:
    global train_instances, test_instances, result_list

    if flag:
        result_list = [[""] * 7 for _ in data_set_full_name_list]
        for count, full_name in enumerate(data_set_full_name_list):
            global data_filename, data_set_name
            data_filename = full_name
            print(data_filename)
            data_set_name = os.path.splitext(os.path.basename(data_filename))[0]
            print(f"{data_set_name},", end="")
            train_instances = load_dataset(data_filename)
            if train_instances is None:
                continue
            run_dsvl()
    else:
        print(os.path.basename(data_filename), end=" ")
        train_instances = load_dataset(data_filename)
        test_instances = load_dataset(data_filename)
        run_dsvl()


def run_dsvl() -> None:
    global runtime_ms, auc_score
    assert train_instances is not None

    begin = _now_ms()
    vc = ValueCentroid()
    cp_list = vc.initial_centroid_list(train_instances)
    cp_list = vc.generate_coupled_centroids(cp_list, train_instances)
    vc.obtain_global_centroid(cp_list, train_instances)
    dsvl = DSVL(cp_list)
    dsvl.value_outlierness_learning(train_instances.num_instances)
    runtime_ms = _now_ms() - begin

    if fs:
        feat_num = int(math.ceil((train_instances.num_attributes() - 1) * 0.5))
        dsvl.feature_selection(train_instances, feat_num, dir_executing, data_set_name)
    else:
        begin = _now_ms()
        outlier_scores = dsvl.scoring_test_instances(train_instances)
        classes = divide(train_instances, invert=True)
        rank_list = rank_instances_based_outlier_scores(outlier_scores, classes)
        save_results(outlier_scores)
        runtime_ms += _now_ms() - begin
        auc_score = compute_auc_according_to_outlier_ranking(classes, rank_list)
        print(f"{format_output(auc_score)},{format_output(runtime_ms / 1000.0)}")


def build_data_sets_path_list(data_set_files_path: str) -> None:
    print(data_set_files_path)
    names = sorted(os.listdir(data_set_files_path))
    full: List[str] = []
    short: List[str] = []
    for name in names:
        lower = name.lower()
        if lower.endswith(".csv") or lower.endswith(".arff"):
            full.append(os.path.join(data_set_files_path, name))
            short.append(os.path.splitext(name)[0])
    global data_set_full_name_list, data_set_name_list
    data_set_full_name_list = full
    data_set_name_list = short


def read_data_set(data_set_file_full_path: str) -> Optional[Dataset]:
    return load_dataset(data_set_file_full_path)


def get_data_set_info(path: str) -> None:
    build_data_sets_path_list(path)
    for i, fullname in enumerate(data_set_full_name_list):
        insts = read_data_set(fullname)
        if insts is None:
            continue
        num_attrs = sum(1 for j in range(insts.num_attributes() - 1) if not insts.attribute(j).is_nominal())
        cat_attrs = sum(1 for j in range(insts.num_attributes() - 1) if insts.attribute(j).is_nominal())
        num_outliers = sum(
            1
            for k in range(insts.num_instances)
            if insts.instance(k).value(insts.num_attributes() - 1) == 0
        )
        outlier_ratio = num_outliers / insts.num_instances if insts.num_instances else 0
        print(
            f"{data_set_name_list[i]},{insts.num_instances},{insts.num_attributes()},"
            f"{num_attrs},{cat_attrs},{outlier_ratio}"
        )


def rank_instances_based_outlier_scores(
    outlier_scores: Dict[int, float],
    classes: Dataset,
) -> "OrderedDict[int, int]":
    items = sorted(outlier_scores.items(), key=lambda x: x[1])
    rank_list: "OrderedDict[int, int]" = OrderedDict()

    for rank, (index, score) in enumerate(items, start=1):
        rank_list[index] = rank
        cl = "1" if classes.instance(index).value(classes.num_attributes() - 1) == 0 else "0"
        _ = score, cl

    return rank_list


def compute_auc_according_to_outlier_ranking(
    classes: Dataset, rank_list: Dict[int, int]
) -> float:
    total_rank = 0
    positive_num = 0
    for i in range(classes.num_instances):
        if classes.instance(i).value(0) == 0:
            total_rank += rank_list[i]
            positive_num += 1
    if positive_num == 0:
        return 0.0
    n = classes.num_instances
    return (total_rank - (positive_num**2 + positive_num) / 2) / (
        positive_num * (n - positive_num)
    )


def format_output(output_value: float) -> str:
    return f"{output_value:.4f}"


def find_indexes_of_value(values: List[float], value: float) -> List[int]:
    return [i for i, v in enumerate(values) if v == value]


def acertou(num_outliers: int, tipo: str, position_ranking: int) -> str:
    if tipo == "I" and position_ranking > num_outliers:
        return "True"
    if tipo == "O" and position_ranking <= num_outliers:
        return "True"
    return "False"


def calculate_ranking(input_list: List[float]) -> List[int]:
    ranking_list = [0] * len(input_list)
    sorted_list = sorted(input_list, reverse=True)
    count = 1
    i = 0
    while i < len(sorted_list):
        val = sorted_list[i]
        ranks = find_indexes_of_value(input_list, val)
        for index in ranks:
            ranking_list[index] = count
        count += len(ranks)
        i += len(ranks)
    return ranking_list


def save_results(scores: Dict[int, float]) -> None:
    assert train_instances is not None
    data_lines: List[List[str]] = [
        ["dataset", "algoritmo", "parameter", "point", "type", "detect", "score", "ranking"]
    ]

    numbers_list = list(scores.values())
    numbers_list.reverse()

    num_outliers = 0
    for idx in range(train_instances.num_instances):
        columns = _instance_as_columns(train_instances, idx)
        tipo = "I" if columns[-1].strip() == "no" else "O"
        if tipo == "O":
            num_outliers += 1

    ranking = calculate_ranking(numbers_list)
    arquivo = os.path.basename(data_filename)

    for idx in range(train_instances.num_instances):
        columns = _instance_as_columns(train_instances, idx)
        tipo = "I" if columns[-1].strip() == "no" else "O"
        data_lines.append(
            [
                arquivo,
                "SDRW",
                "",
                str(idx + 1),
                tipo,
                acertou(num_outliers, tipo, ranking[idx]),
                str(numbers_list[idx]),
                str(ranking[idx]),
            ]
        )

    csv = CSV()
    try:
        csv.save_results(DIR_RESULT, arquivo, data_lines)
    except OSError as exc:
        print(exc, file=sys.stderr)


def _instance_as_columns(data: Dataset, index: int) -> List[str]:
    inst = data.instance(index)
    parts = []
    for i in range(data.num_attributes()):
        attr = data.attribute(i)
        val = int(inst.value(i))
        if attr.is_nominal() and val < len(attr.values):
            parts.append(attr.values[val])
        else:
            parts.append(str(val))
    return parts


def _now_ms() -> int:
    return int(time.time() * 1000)
