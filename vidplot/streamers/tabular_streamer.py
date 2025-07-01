import pandas as pd
from typing import Any, Dict, Iterable, Optional, Tuple, Union, List
from vidplot.core import DataStreamer, KnownDurationProtocol
from vidplot.streamers.utils import _load_and_validate_data_source


class TabularStreamer(DataStreamer, KnownDurationProtocol):
    """
    A tabular data streamer that reads time-series data from various sources.
    Uses a sliding-window buffer for nearest-time or LOCF sampling, similar to video streaming.
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
    ) -> None:
        # Load and validate the tabular source
        self._timestamps, self._data = _load_and_validate_data_source(
            data_source, data_col, time_col
        )
        if abs(self._timestamps[0] - 0) >= tol:
            raise ValueError(f"Expected first timestamp close to 0, got {self._timestamps[0]:.5f}")
        
        if sample_rate is None:
            assert len(self._timestamps) >= 2, "At least 2 data points are required if sample rate is not provided."
            sample_rate = (self._timestamps[-1] - self._timestamps[0]) / (len(self._timestamps) - 1)

        super().__init__(name=name, sample_rate=sample_rate)

        # Determine duration
        if duration is not None:
            assert duration > 0, "Duration must be positive."
            self._duration = duration
        else:
            self._duration = self._timestamps[-1]

        # Validate streaming method
        assert stream_method in [
            "nearest_neighbor",
            "LOCF",
        ], f"Unsupported stream method '{stream_method}'"
        self._stream_method = stream_method
        self._tol = tol

        # Internal buffer and seeker index
        self._seek_idx = 0
        self._buf: List[Tuple[float, Any]] = []  # holds up to 2 (timestamp, data) pairs

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def approx_duration(self) -> float:
        return self._duration

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "data_points": len(self._data),
            "time_range": (self._timestamps[0], self._duration),
            "stream_method": self._stream_method,
        }

    def _seek(self) -> Tuple[float, Any]:
        # Advance index and read next timestamp/data
        if self._seek_idx >= len(self._timestamps):
            raise StopIteration
        ts = self._timestamps[self._seek_idx]
        payload = self._data[self._seek_idx]
        self._seek_idx += 1
        return ts, payload

    def stream(self) -> Any:
        target_time = self._clock

        # Fill buffer until it brackets or passes target_time
        while not self._buf or self._buf[-1][0] < target_time - self._tol:
            try:
                ts, payload = self._seek()
                if len(self._buf) == 2:
                    self._buf.pop(0)
                self._buf.append((ts, payload))
            except StopIteration:
                break

        # If buffer has data at or beyond target, pick correct sample
        if self._buf and self._buf[-1][0] >= target_time - self._tol:
            if len(self._buf) == 1:
                return self._buf[0][1]
            t0, d0 = self._buf[0]
            t1, d1 = self._buf[1]
            if self._stream_method == "nearest_neighbor":
                return d0 if abs(t0 - target_time) < abs(t1 - target_time) else d1
            # LOCF: last observation carried forward
            return d0

        # If we ran out of data but still within duration, return last-known
        if target_time <= self._duration and self._buf:
            return self._buf[-1][1]

        # Otherwise, end of stream
        raise StopIteration
