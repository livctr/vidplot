import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, Optional, Union

from vidplot.core import StaticDataStreamer, KnownDurationProtocol
from vidplot.streamers.utils import _load_and_validate_data_source


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
        sample_rate: Optional[float] = None,
        duration: Optional[float] = None,
        stream_method: str = "nearest_neighbor",
        tol: float = 1e-3,
        num_samples: int = 1000,
    ) -> None:
        self._timestamps, self._data = _load_and_validate_data_source(data_source, data_col, time_col)
        if abs(self._timestamps[0] - 0) >= tol:
            raise ValueError(
                f"Expected the first timestamp entry to be close to 0. Instead got "
                f"{self._timestamps[0]:.5f}"
            )

        if sample_rate is None:
            assert len(self._timestamps) >= 2, "At least 2 data points are required if sample rate is not provided."
            sample_rate = (self._timestamps[-1] - self._timestamps[0]) / (len(self._timestamps) - 1)
        
        if duration is None:
            duration = self._timestamps[-1]
        
        # Subsample the data
        self._timestamps = np.array(self._timestamps)
        evenly_spaced_timestamps = np.linspace(self._timestamps[0], duration, num_samples, endpoint=False)
        idxs = np.searchsorted(self._timestamps - tol, evenly_spaced_timestamps, side='left') - 1
        data = [self._data[i] for i in idxs]

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
