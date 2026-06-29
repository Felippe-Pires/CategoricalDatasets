"""Load categorical datasets (ARFF/CSV) for the SDRW algorithm."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np


@dataclass
class Attribute:
    name: str
    values: List[str] = field(default_factory=list)

    def is_nominal(self) -> bool:
        return len(self.values) > 0

    def num_values(self) -> int:
        if self.values:
            return len(self.values)
        return 0


@dataclass
class Instance:
    values: np.ndarray

    def value(self, index: int) -> float:
        return float(self.values[index])


@dataclass
class Dataset:
    """Minimal Weka-like Instances wrapper for SDRW."""

    attributes: List[Attribute]
    data: np.ndarray
    class_index: int = -1

    @property
    def num_instances(self) -> int:
        return int(self.data.shape[0])

    def num_attributes(self) -> int:
        return len(self.attributes)

    def attribute(self, index: int) -> Attribute:
        return self.attributes[index]

    def instance(self, index: int) -> Instance:
        return Instance(self.data[index])

    def set_class_index(self, index: int) -> None:
        self.class_index = index


def _decode_arff_value(raw) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def load_dataset(path: str) -> Optional[Dataset]:
    """Read an ARFF or CSV file into a Dataset."""
    path = os.path.abspath(path)
    lower = path.lower()
    if lower.endswith(".arff"):
        return _load_arff(path)
    if lower.endswith(".csv"):
        return _load_csv(path)
    return None


def _load_arff(path: str) -> Optional[Dataset]:
    try:
        from scipy.io import arff
    except ImportError:
        raise ImportError("scipy is required to load ARFF files (pip install scipy)")

    raw_data, meta = arff.loadarff(path)
    names = list(meta.names())
    attributes: List[Attribute] = []
    columns = []

    for name in names:
        field = meta[name]
        if hasattr(field, "values"):
            values = [_decode_arff_value(v) for v in field.values]
            attributes.append(Attribute(name, values))
            col = raw_data[name]
            if col.dtype.kind in ("S", "O", "U"):
                decoded = np.array([_decode_arff_value(v) for v in col], dtype=object)
                index_map = {v: i for i, v in enumerate(values)}
                columns.append(np.array([index_map[v] for v in decoded], dtype=float))
            else:
                columns.append(np.asarray(col, dtype=float))
        else:
            attributes.append(Attribute(name, []))
            columns.append(np.asarray(raw_data[name], dtype=float))

    data = np.column_stack(columns)
    dataset = Dataset(attributes, data)
    dataset.set_class_index(dataset.num_attributes() - 1)
    return dataset


# Columns with at most this many distinct values are treated as categorical
# (e.g. binary 0/1 flags in covertype.csv). High-cardinality numerics stay numeric.
_MAX_NOMINAL_CARDINALITY = 256


def _csv_column_as_nominal(series) -> bool:
    import pandas as pd

    if series.dtype == object or str(series.dtype).startswith("category"):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return int(series.nunique(dropna=True)) <= _MAX_NOMINAL_CARDINALITY
    return False


def _nominal_values_and_indices(series):
    """Build category labels and 0-based index column for a nominal series."""
    import pandas as pd

    if pd.api.types.is_numeric_dtype(series):
        uniq_vals = sorted(series.dropna().unique().tolist())
        labels = []
        for v in uniq_vals:
            fv = float(v)
            labels.append(str(int(fv)) if fv == int(fv) else str(v))
        value_to_index = {float(v): i for i, v in enumerate(uniq_vals)}
        indices = series.map(lambda x: value_to_index[float(x)] if pd.notna(x) else np.nan)
    else:
        labels = sorted(series.dropna().astype(str).unique().tolist())
        value_to_index = {v: i for i, v in enumerate(labels)}
        indices = series.astype(str).map(value_to_index)

    return labels, indices.astype(float).to_numpy()


def _load_csv(path: str, delimiter: str = ",") -> Optional[Dataset]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to load CSV files (pip install pandas)")

    df = pd.read_csv(path)
    attributes: List[Attribute] = []
    columns = []

    for col_name in df.columns:
        series = df[col_name]
        if _csv_column_as_nominal(series):
            labels, indices = _nominal_values_and_indices(series)
            attributes.append(Attribute(str(col_name), labels))
            columns.append(indices)
        else:
            attributes.append(Attribute(str(col_name), []))
            columns.append(series.astype(float).to_numpy())

    data = np.column_stack(columns)
    dataset = Dataset(attributes, data)
    dataset.set_class_index(dataset.num_attributes() - 1)
    return dataset


def divide(instances: Dataset, invert: bool = False) -> Dataset:
    """Split off the class attribute (mirrors DSVL4ODUtils.divide)."""
    if instances.class_index < 0:
        raise ValueError("A class attribute has to be specified.")

    if invert:
        idx = instances.class_index
        return Dataset(
            [instances.attributes[idx]],
            instances.data[:, idx : idx + 1].copy(),
            class_index=0,
        )

    attrs = [a for i, a in enumerate(instances.attributes) if i != instances.class_index]
    cols = [i for i in range(instances.num_attributes()) if i != instances.class_index]
    subset = Dataset(attrs, instances.data[:, cols].copy(), class_index=-1)
    return subset
