import math
from typing import Any, Dict, Mapping


class DiskAnnParams:
    _defaults: Dict[str, Any] = {
        "diskann:space": "l2",
        "diskann:graph_degree": 32,
        "diskann:build_search_list_size": 64,
        "diskann:search_list_size": 64,
        "diskann:beam_width": 4,
        "diskann:rebuild_threshold": 1000,
        "diskann:pq_bytes": 8,
        "diskann:num_threads": 4,
        "diskann:alpha": 1.2,
    }

    def __init__(self, metadata: Mapping[str, Any]):
        values = {**self._defaults, **self.extract(metadata)}
        self.space = str(values["diskann:space"])
        self.graph_degree = int(values["diskann:graph_degree"])
        self.build_search_list_size = int(values["diskann:build_search_list_size"])
        self.search_list_size = int(values["diskann:search_list_size"])
        self.beam_width = int(values["diskann:beam_width"])
        self.rebuild_threshold = int(values["diskann:rebuild_threshold"])
        self.pq_bytes = int(values["diskann:pq_bytes"])
        self.num_threads = int(values["diskann:num_threads"])
        self.alpha = float(values["diskann:alpha"])

    @classmethod
    def extract(cls, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        selected = {
            key: value for key, value in metadata.items() if key.startswith("diskann:")
        }
        for key, value in selected.items():
            if key not in cls._defaults:
                raise ValueError(f"Unknown DiskANN parameter: {key}")
            if key == "diskann:space":
                valid = isinstance(value, str) and value in ("l2", "cosine")
            elif key == "diskann:alpha":
                valid = (
                    type(value) in (int, float)
                    and math.isfinite(value)
                    and value >= 1.0
                )
            else:
                valid = type(value) is int and value > 0
            if not valid:
                raise ValueError(
                    f"Invalid value for DiskANN parameter: {key} = {value}"
                )
        values = {**cls._defaults, **selected}
        if values["diskann:build_search_list_size"] < values["diskann:graph_degree"]:
            raise ValueError("DiskANN build_search_list_size must be >= graph_degree")
        if values["diskann:beam_width"] > 128:
            raise ValueError("DiskANN beam_width must be <= 128")
        return selected
