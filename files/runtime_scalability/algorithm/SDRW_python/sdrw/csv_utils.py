"""CSV result export (port of edu.uts.aai.utils.CSV)."""

from __future__ import annotations

import os
import re
from typing import List, Sequence


# Java's "\\R" (any line break); Python re has no \R — use explicit line-break class
_LINE_BREAKS = re.compile(r"[\r\n\v\f\u0085\u2028\u2029]+")


class CSV:
    def __init__(self, algorithm: str = "SDRW") -> None:
        self.algorithm = algorithm

    @staticmethod
    def escape_special_characters(data: str) -> str:
        if data is None:
            raise ValueError("Input data cannot be null")
        text = str(data)
        escaped = _LINE_BREAKS.sub(" ", text)
        if ";" in text or '"' in text or "'" in text:
            text = text.replace('"', '""')
            escaped = f'"{text}"'
        return escaped

    def convert_to_csv(self, row: Sequence[str]) -> str:
        return ";".join(self.escape_special_characters(str(cell)) for cell in row)

    def save_results(self, path: str, filename: str, data_lines: List[List[str]]) -> None:
        out_dir = os.path.join(path, self.algorithm)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, filename)
        with open(out_file, "w", encoding="utf-8", newline="") as f:
            for row in data_lines:
                f.write(self.convert_to_csv(row) + "\n")
