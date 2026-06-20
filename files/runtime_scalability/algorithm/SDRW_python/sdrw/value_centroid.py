"""Coupled centroids across categorical feature values."""

from __future__ import annotations

from typing import ClassVar, List, Optional

from .dataset import Dataset
from .value_node import ValueNode


class ValueCentroid:
    global_centroid: ClassVar[Optional[ValueNode]] = None

    def __init__(self) -> None:
        self.org_feat = 0
        self.cen_list: List[ValueNode] = []

    def initial_centroid_list(self, data: Dataset) -> List["ValueCentroid"]:
        cp_list: List[ValueCentroid] = []
        d = data.num_attributes() - 1
        for i in range(d):
            cp = ValueCentroid()
            cp.org_feat = i
            card = data.attribute(i).num_values()
            for k in range(card):
                cen = ValueNode(i, k, data)
                values = data.attribute(i).values
                label = values[k] if k < len(values) else str(k)
                cen.set_categorical_content(label)
                cp.cen_list.append(cen)
            cp_list.append(cp)
        return cp_list

    def generate_coupled_centroids(
        self, cp_list: List["ValueCentroid"], data: Dataset
    ) -> List["ValueCentroid"]:
        for i, cp in enumerate(cp_list):
            for j in range(data.num_instances):
                inst = data.instance(j)
                index = int(inst.value(i))
                cen = cp.cen_list[index]
                cen.update_centroid(inst)
        return cp_list

    def print_coupled_patterns(self, cp_list: List["ValueCentroid"], attr_id: int) -> None:
        cp = cp_list[attr_id]
        for i, cen in enumerate(cp.cen_list):
            print(f"No.{i}:", end="")
            cen.print_centroid_info()
            print()

    def obtain_global_centroid(self, cp_list: List["ValueCentroid"], data: Dataset) -> None:
        ValueCentroid.global_centroid = ValueNode.global_template(data)
        cp = cp_list[0]
        for cen in cp.cen_list:
            ValueCentroid.global_centroid.generate_global_centroid(cen)

    def get_cen_list(self) -> List[ValueNode]:
        return self.cen_list

    def get_global_centroid(self) -> ValueNode:
        assert ValueCentroid.global_centroid is not None
        return ValueCentroid.global_centroid
