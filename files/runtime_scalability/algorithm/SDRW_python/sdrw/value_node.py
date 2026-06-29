"""Value-level centroid for categorical outlier detection."""

from __future__ import annotations

import math
from typing import List, Optional

from .dataset import Dataset, Instance
from .feature_info import FeatureInfo


class ValueNode:
    def __init__(
        self,
        attr_id: int = -1,
        value_id: int = -1,
        insts: Optional[Dataset] = None,
    ) -> None:
        self.org_feat = attr_id
        self.value_index = value_id
        self.attr_list: List[FeatureInfo] = []
        self.most_coupled_feat = -1
        self.outlier_score = 0.0
        self.intra_od = -1.0
        self.uncertainty: Optional[List[float]] = None
        self.weight = 0.0
        self.weighted_degree = 0.0
        self.content = ""

        if insts is not None:
            self.dim = insts.num_attributes() - 1
            self.size = insts.num_instances
            for i in range(self.dim):
                card = insts.attribute(i).num_values()
                self.attr_list.append(FeatureInfo(i, card))
        else:
            self.dim = 0
            self.size = 0

    @classmethod
    def global_template(cls, insts: Dataset) -> "ValueNode":
        node = cls()
        node.dim = insts.num_attributes() - 1
        node.size = insts.num_instances
        node.uncertainty = [0.0] * node.dim
        for i in range(node.dim):
            card = insts.attribute(i).num_values()
            node.attr_list.append(FeatureInfo(i, card))
        return node

    def update_centroid(self, inst: Instance) -> None:
        for i in range(self.dim):
            ai = self.attr_list[i]
            index = int(inst.value(i))
            ai.add_freq(index)

    def generate_global_centroid(self, local_centd: "ValueNode") -> None:
        for i in range(self.dim):
            ai1 = local_centd.attr_list[i]
            gai1 = self.attr_list[i]
            set_size1 = ai1.num_of_value()
            for vid in range(set_size1):
                gai1.add_freq(vid, ai1.value(vid))

            u = 0.0
            if set_size1 == 1:
                self.set_uncertainty(i, -u)
                continue

            for vid in range(set_size1):
                g_freq = self.global_freq(i, vid)
                p = g_freq / self.size if self.size else 0.0
                tmp = 0.0
                if p != 0:
                    tmp = p * math.log10(p)
                denom = math.log10(set_size1) / math.log10(2)
                u += tmp / denom if denom else 0.0
            self.set_uncertainty(i, -u)

    def global_freq(self, attr_id: int, value_id: int) -> float:
        return self.attr_list[attr_id].value(value_id)

    def print_centroid_info(self) -> None:
        for ai in self.attr_list:
            ai.print_attribute_info()
            print("##", end="")
        print("\r\n")
        if self.uncertainty is not None:
            print(",".join(str(u) for u in self.uncertainty))
        print("\r\n")

    def set_outlier_score(self, od: float) -> None:
        self.outlier_score = od

    def add_outlier_score(self, od: float) -> None:
        self.outlier_score += od

    def get_outlier_score(self) -> float:
        return self.outlier_score

    def set_intra_od(self, od: float) -> None:
        self.intra_od = od

    def get_intra_od(self) -> float:
        return self.intra_od

    def set_uncertainty(self, index: int, u: float) -> None:
        assert self.uncertainty is not None
        self.uncertainty[index] = u

    def get_uncertainty(self, index: int) -> float:
        assert self.uncertainty is not None
        return self.uncertainty[index]

    def set_categorical_content(self, content: str) -> None:
        self.content = content

    def get_categorical_content(self) -> str:
        return self.content

    def get_weighted_degree(self) -> float:
        return self.weighted_degree

    def set_weighted_degree(self, weighted_degree: float) -> None:
        self.weighted_degree = weighted_degree

    def get_weight(self) -> float:
        return self.weight

    def set_weight(self, weight: float) -> None:
        self.weight = weight

    def __str__(self) -> str:
        return f"{self.org_feat}_{self.value_index}"
