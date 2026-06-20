"""Random two-letter label mapping for categorical values."""

from __future__ import annotations

import random
import string
from typing import Dict, List


def generate_random_label(rng: random.Random | None = None) -> str:
    source = rng or random
    letters = string.ascii_lowercase
    return "".join(source.choice(letters) for _ in range(2))


def map_column_to_random_labels(values: List[str]) -> List[str]:
    label_map: Dict[str, str] = {}
    for label in set(values):
        label_map[label] = generate_random_label()
    return [label_map[v] for v in values]
