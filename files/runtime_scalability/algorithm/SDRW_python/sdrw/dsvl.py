"""
DSVL / SDRW core: intra-value outlierness, value graph, dense subgraph discovery.
Python port of edu.uts.aai.utils.DSVL (Java).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

from .dataset import Dataset
from .plot import plot_y_points
from .value_centroid import ValueCentroid
from .value_node import ValueNode


class DSVL:
    def __init__(self, cp_list: List[ValueCentroid]) -> None:
        self.cp_list = cp_list
        self.dim = len(cp_list)
        self.v_matrix: List[List[float]] = []
        self.val_feat_ids: List[int] = []
        self.val_identifier: List[str] = []
        self.val_ind_wgts: List[float] = []
        self.attr_weight: Optional[List[float]] = None

    def dense_subgraph_discovery(self, data_size: int) -> str:
        self.calc_intra_feature_value_outlierness(data_size)
        self.value_adjacency_matrix()
        discard_vals: List[str] = []
        den = self.charikar_greedy_search_for_value_graph(discard_vals)
        max_den = float("-inf")
        max_id = -1
        for i, d in enumerate(den):
            if d > max_den:
                max_den = d
                max_id = i
        print(f"MAX:{max_den:.6f} ")
        from . import dsvl4od_utils

        plot_y_points(
            den,
            3,
            dsvl4od_utils.data_set_name,
            dsvl4od_utils.data_set_name,
            "Iteration",
            "Avg. Incoming Edge Weight",
        )
        return discard_vals[max_id]

    def value_outlierness_learning(self, data_size: int) -> None:
        self.calc_intra_feature_value_outlierness(data_size)
        self.value_adjacency_matrix()
        discard_vals: List[str] = []
        self.charikar_greedy_search_for_value_graph(discard_vals)
        self.weighted_degree2_score()

    def calc_intra_feature_value_outlierness(self, data_size: int) -> None:
        m_freq = [0.0] * self.dim
        for i, cp in enumerate(self.cp_list):
            max_freq = 0.0
            for j, cen in enumerate(cp.cen_list):
                global_freq = cen.global_freq(i, j)
                if global_freq > max_freq:
                    max_freq = global_freq
            m_freq[i] = max_freq

        for i, cp in enumerate(self.cp_list):
            for j, cen in enumerate(cp.cen_list):
                global_freq = cen.global_freq(i, j)
                if global_freq == 0:
                    continue
                intra = (
                    abs(global_freq - m_freq[i]) / m_freq[i]
                    + (1 - m_freq[i] / data_size)
                ) / 2.0
                cen.set_intra_od(intra)

    def value_adjacency_matrix(self) -> None:
        print()
        for i in range(self.dim):
            cp = self.cp_list[i]
            for j, cen in enumerate(cp.cen_list):
                if cen.global_freq(i, j) == 0:
                    print("0.0 ", end="")
                    continue

                col: List[float] = []
                tmp = 0.0
                for k in range(self.dim):
                    ai = cen.attr_list[k]
                    gai = cp.get_global_centroid().attr_list[k]
                    length = ai.num_of_value()
                    for l in range(length):
                        if k == cen.org_feat and gai.value(l) != 0:
                            col.append(0.0)
                            print("0.0 ", end="")
                            continue

                        freq = ai.value(l)
                        g_freq = gai.value(l)
                        cen_freq = cen.global_freq(i, j)
                        if cen_freq != 0 and g_freq != 0:
                            w = (
                                cen.get_intra_od()
                                * self.cp_list[k].cen_list[l].get_intra_od()
                                * (freq / (cen_freq * g_freq))
                            )
                            col.append(w)
                            tmp += w
                            print(f"{w} ", end="")
                        else:
                            print("0.0 ", end="")

                self.v_matrix.append(col)
                self.val_feat_ids.append(i)
                self.val_ind_wgts.append(tmp)
                self.val_identifier.append(f"{i}_{j}")
                cen.set_weighted_degree(tmp)
                print()

    def update_weighted_degree(self) -> None:
        for i in range(self.dim):
            cp = self.cp_list[i]
            for j, cen in enumerate(cp.cen_list):
                if cen.global_freq(i, j) == 0:
                    continue
                tmp = 0.0
                for k in range(self.dim):
                    ai = cen.attr_list[k]
                    gai = cp.get_global_centroid().attr_list[k]
                    length = ai.num_of_value()
                    for l in range(length):
                        if k == cen.org_feat and gai.value(l) != 0:
                            continue
                        freq = ai.value(l)
                        g_freq = gai.value(l)
                        cen_freq = cen.global_freq(i, j)
                        if cen_freq != 0 and g_freq != 0:
                            w = (
                                cen.get_weight()
                                * self.cp_list[k].cen_list[l].get_weight()
                                * (freq / (cen_freq * g_freq))
                            )
                            tmp += w
                cen.weighted_degree = tmp

    def weighted_degree2_score(self) -> None:
        self.weight_normalization()
        self.update_weighted_degree()

        self.attr_weight = [1.0] * self.dim
        for i in range(self.dim):
            cp = self.cp_list[i]
            for j, cen in enumerate(cp.cen_list):
                prob = cen.get_weight()
                cen.set_outlier_score(prob)
                print(f"{prob},", end="")
                self.attr_weight[i] *= 1 - prob

        print()
        self.attr_weight = [1 - w for w in self.attr_weight]

    def weight_normalization(self) -> None:
        total_wgt = sum(cen.get_weight() for cp in self.cp_list for cen in cp.cen_list)

        for cp in self.cp_list:
            for cen in cp.cen_list:
                cen.set_weight(cen.get_weight() / total_wgt if total_wgt else 0.0)

    def scoring_test_instances(self, data: Dataset) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        assert self.attr_weight is not None

        for j in range(data.num_instances):
            inst = data.instance(j)
            score = 1.0
            for i in range(self.dim):
                cp = self.cp_list[i]
                index = int(inst.value(i))
                cen = cp.cen_list[index]
                s = cen.get_outlier_score()
                score *= math.pow(1 - s, self.attr_weight[i])
            score = 1 - score
            print(score)
            scores[j] = score

        return scores

    def charikar_greedy_search_for_value_graph(
        self, discard_vals: List[str]
    ) -> List[float]:
        count = len(self.val_ind_wgts)
        edge_weights = list(self.val_ind_wgts)
        v_matrix = [list(row) for row in self.v_matrix]
        identifiers = list(self.val_identifier)

        den: List[float] = []
        id_ = 0
        accumulate_den = 0.0
        sb: List[str] = []

        while count > 0:
            density = self._compute_density(edge_weights, count)
            den.append(density)
            id_ += 1
            accumulate_den += density
            discard_vals.append(",".join(sb))

            min_w = float("inf")
            mid = -1
            for i, w in enumerate(edge_weights):
                if w < min_w:
                    min_w = w
                    mid = i

            self._remove_one_feature_value(mid, edge_weights, v_matrix)
            str_id = identifiers.pop(mid)
            sb.append(f"{str_id},")
            feat_id, value_id = (int(x) for x in str_id.split("_"))
            cen = self.cp_list[feat_id].cen_list[value_id]
            cen.set_weight(accumulate_den)
            count -= 1

        return den

    def _remove_one_feature_value(
        self,
        vid: int,
        edge_weights: List[float],
        v_matrix: List[List[float]],
    ) -> None:
        v_matrix.pop(vid)
        edge_weights.pop(vid)
        for k in range(len(v_matrix)):
            col = v_matrix[k]
            w = col.pop(vid)
            edge_weights[k] -= w

    @staticmethod
    def _compute_density(edge_weights: List[float], val_num: int) -> float:
        density = sum(edge_weights)
        return density / (2 * val_num) if val_num else 0.0

    def feature_selection(
        self, data: Dataset, feat_num: int, path: str, name: str
    ) -> None:
        attr_id_array = list(range(self.dim))
        assert self.attr_weight is not None
        attr_id_array.sort(key=lambda idx: self.attr_weight[idx])

        for i, attr_id in enumerate(attr_id_array):
            print(f"{attr_id}:{self.attr_weight[attr_id]},")
        print()

        count = len(attr_id_array)
        remove_ids: List[int] = []
        i = 0
        while count > feat_num:
            remove_ids.append(attr_id_array[i] + 1)
            i += 1
            count -= 1

        new_data = _remove_attributes(data, remove_ids)
        out_path = os.path.join(path, f"{name}_{count:02d}.arff")
        _write_arff(new_data, out_path)
        print(",".join(str(x) for x in remove_ids))


def _remove_attributes(data: Dataset, one_based_indices: List[int]) -> Dataset:
    """Remove attributes by 1-based indices (Weka Remove convention)."""
    remove_set = {i - 1 for i in one_based_indices}
    keep = [i for i in range(data.num_attributes()) if i not in remove_set]
    attrs = [data.attributes[i] for i in keep]
    matrix = data.data[:, keep].copy()
    result = Dataset(attrs, matrix, class_index=-1)
    if data.class_index in keep:
        result.set_class_index(keep.index(data.class_index))
    elif len(keep) > 0:
        result.set_class_index(len(keep) - 1)
    return result


def _write_arff(data: Dataset, path: str) -> None:
    import os

    lines = ["@relation SDRW_subset", ""]
    for attr in data.attributes:
        if attr.is_nominal():
            values = ",".join(f'"{v}"' for v in attr.values)
            lines.append(f"@attribute {attr.name} {{{values}}}")
        else:
            lines.append(f"@attribute {attr.name} numeric")
    lines.append("")
    lines.append("@data")
    for row in data.data:
        parts = []
        for i, val in enumerate(row):
            if data.attributes[i].is_nominal():
                parts.append(data.attributes[i].values[int(val)])
            else:
                parts.append(str(val))
        lines.append(",".join(parts))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
