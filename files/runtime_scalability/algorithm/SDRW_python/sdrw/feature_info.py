"""Per-feature frequency storage for value centroids."""

from __future__ import annotations

from typing import List


class FeatureInfo:
    def __init__(self, index: int, length: int = 0) -> None:
        self.index = index
        if length > 0:
            self.freq_array = [0.0] * length
            self.w_array = [0.0] * length
        else:
            self.freq_array: List[float] = []
            self.w_array: List[float] = []

    def add_freq(self, index: int, freq: float = 1.0) -> None:
        self.freq_array[index] += freq

    def set_freq(self, index: int, freq: float) -> None:
        self.freq_array[index] = freq

    def value(self, index: int) -> float:
        return self.freq_array[index]

    def set_weight(self, index: int, weight: float) -> None:
        self.w_array[index] = weight

    def get_weight(self, attr_id: int) -> float:
        return self.w_array[attr_id]

    def add_weight(self, index: int, weight: float) -> None:
        self.w_array[index] += weight

    def weight_value(self, index: int) -> float:
        return self.w_array[index]

    def num_of_value(self) -> int:
        return len(self.freq_array)

    def num_non_zero_freq(self) -> int:
        return sum(1 for f in self.freq_array if f >= 1)

    def print_attribute_info(self) -> None:
        print(",".join(str(f) for f in self.freq_array), end="")
