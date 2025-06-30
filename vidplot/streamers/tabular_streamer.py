import pandas as pd
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from vidplot.core import DataStreamer, KnownDurationProtocol
from vidplot.streamers.utils import _stream_with_last_frame_handling, _load_and_validate_data_source


class TabularStreamer(DataStreamer, KnownDurationProtocol):
    """
    A tabular data streamer that reads time-series data from various sources.
    Uses a sliding-window logic for nearest-time sampling, similar to video streaming.
    """

    def __init__(
        self,
        name: str,
        data_source: Union[pd.DataFrame, str, Dict[str, Iterable]],
        data_col: str,
        time_col: str,
        sample_rate: float = 30.0,
        stream_method: str = "nearest",
        increase_endpoint: bool = True,
    ):
        super().__init__(name=name, sample_rate=sample_rate)
        self._timestamps, self._data = _load_and_validate_data_source(
            data_source, data_col, time_col
        )
        if abs(self._timestamps[0] - 0) > 1e-5:
            raise ValueError(
                f"Expected the first timestamp entry to be close to 0. Instead got"
                f" {self._timestamps[0]:.5f}"
            )

        if len(self._timestamps) >= 2:
            timestep = float(self._timestamps[-1] - self._timestamps[0]) / (
                len(self._timestamps) - 1
            )
            self._duration = (
                self._timestamps[-1] + timestep if increase_endpoint else self._timestamps[-1]
            )
        else:
            self._duration = 0.0
        self._stream_method = stream_method.lower()
        if self._stream_method not in [
            "nearest",
            "nearest_left",
            "nearest_right",
        ]:
            raise ValueError(
                f"Invalid stream_method: {self._stream_method}."
                "Must be 'nearest', 'nearest_left', or 'nearest_right'."
            )
        self._prev_idx: Optional[int] = None
        self._cur_idx: Optional[int] = None
        self._prev_ts: Optional[float] = None
        self._cur_ts: Optional[float] = None
        self._prev_data: Optional[Any] = None
        self._cur_data: Optional[Any] = None
        self._last_frame_time: Optional[float] = None
        self._last_frame: Optional[Any] = None

    @property
    def duration(self) -> float:
        """Duration is the last timestamp plus one timestep."""
        return self._duration

    @property
    def approx_duration(self) -> float:
        """Approx duration is just the duration."""
        return self._duration

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "data_points": len(self._data),
            "time_range": (
                self._timestamps[0],
                self._timestamps[0] + self._duration,
            ),
            "stream_method": self._stream_method,
        }

    def _seek(self) -> Tuple[float, Any]:
        if self._cur_idx is None:
            self._cur_idx = 0
        else:
            self._cur_idx += 1
        if self._cur_idx >= len(self._timestamps):
            raise StopIteration
        ts = self._timestamps[self._cur_idx]
        data = self._data[self._cur_idx]
        return ts, data

    def stream(self) -> Any:
        target_time = self._clock

        # before first timestamp
        if target_time <= self._timestamps[0]:
            return self._data[0]

        # after last valid time (allow last frame continuation)
        if target_time > self.duration:
            raise StopIteration

        (
            data,
            self._prev_ts,
            self._prev_data,
            self._cur_ts,
            self._cur_data,
            self._last_frame_time,
            self._last_frame,
        ) = _stream_with_last_frame_handling(
            target_time,
            self._prev_ts,
            self._prev_data,
            self._cur_ts,
            self._cur_data,
            self._last_frame_time,
            self._last_frame,
            self.sample_rate,
            self._seek,
            self._stream_method,
        )
        return data
