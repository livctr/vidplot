import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional, Union
from pathlib import Path

from . import DataStreamer, KnownDurationProtocol, StaticMixin
from .tabular_streamer import _load_and_validate_data_source

class StaticTabularStreamer(DataStreamer, StaticMixin, KnownDurationProtocol):
    """
    A static tabular data streamer that returns the fixed table per call.
    Inherits from StaticMixin for infinite duration behavior.
    """
    def __init__(
        self,
        name: str,
        data_source: Union[pd.DataFrame, str, Dict[str, Iterable]],
        data_col: str,
        time_col: str,
        sample_rate: float = 30.0,
        num_samples: Optional[int] = None,
        subsample_method: str = "nearest"
    ):
        super().__init__(name=name, sample_rate=sample_rate)
        timestamps, data = _load_and_validate_data_source(data_source, data_col, time_col)
        if len(timestamps) >= 2:
            timestep = float(timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
            self._duration = timestamps[-1] + timestep
        else:
            self._duration = 0.

        # Subsample if requested
        if num_samples and num_samples < len(timestamps):
            time_samples = np.linspace(timestamps[0], timestamps[-1], num_samples)
            data = np.array(data)
            if np.issubdtype(data.dtype, np.number):
                # Numeric interpolation
                subsampled_data = np.interp(time_samples, timestamps, data).tolist()
            else:
                # Nearest neighbor for non-numeric data
                idxs = np.searchsorted(timestamps, time_samples, side="left")
                idxs = np.clip(idxs, 0, len(data)-1)
                subsampled_data = [data[i] for i in idxs]
            self._data = subsampled_data
        else:
            self._data = data

    @property
    def duration(self) -> float:
        """Duration is the last timestamp plus one timestep."""
        return self._duration

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "data_points": len(self._data),
            "static": True,
            "data_type": "tabular",
        }

    def _generate_data(self) -> Any:
        """Return all data points as a list."""
        return self._data 
