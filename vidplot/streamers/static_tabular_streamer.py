import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, Optional, Union

from vidplot.core import StaticDataStreamer, KnownDurationProtocol
from .tabular_streamer import _load_and_validate_data_source


class StaticTabularStreamer(StaticDataStreamer, KnownDurationProtocol):
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
        subsample_method: str = "nearest",
    ):
        timestamps, data = _load_and_validate_data_source(data_source, data_col, time_col)
        if abs(timestamps[0] - 0) > 1e-5:
            raise ValueError(
                f"Expected the first timestamp entry to be close to 0. Instead got "
                f"{timestamps[0]:.5f}"
            )
        if len(timestamps) >= 2:
            timestep = float(timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
            duration = timestamps[-1] + timestep
        else:
            duration = 0.0

        # Subsample if requested
        if num_samples and num_samples < len(timestamps):
            time_samples = np.linspace(timestamps[0], duration, num_samples, endpoint=False)
            data = np.array(data)
            if np.issubdtype(data.dtype, np.number):
                # Numeric interpolation
                subsampled_data = np.interp(time_samples, timestamps, data).tolist()
            else:
                # Nearest neighbor for non-numeric data
                idxs = np.searchsorted(timestamps, time_samples, side="left")
                idxs = np.clip(idxs, 0, len(data) - 1)
                subsampled_data = [data[i] for i in idxs]
            data = subsampled_data

        super().__init__(name=name, data=data, sample_rate=sample_rate)
        self._duration = duration

    @property
    def duration(self) -> float:
        """
        The duration used by other streamers for their own calculations.
        """
        return self._duration

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "data_points": len(self._data),
            "static": True,
            "data_type": "tabular",
        }
