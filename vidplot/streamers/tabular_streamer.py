import numpy as np
import pandas as pd
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from pathlib import Path
from vidplot.core import DataStreamer, KnownDurationProtocol
from .utils import _stream_with_last_frame_handling


def _load_and_validate_data_source(
    data_source: Union[pd.DataFrame, str, Dict[str, Any]],
    data_col: str,
    time_col: str,
) -> Tuple[List[float], List[Any]]:
    """
    Loads time-series data from various formats and validates required columns.
    """
    if not data_col:
        raise ValueError("`data_col` must be specified.")
    if not time_col:
        raise ValueError("`time_col` must be specified.")

    def sort_by_time(timestamps: Iterable, values: Iterable):
        if len(timestamps) == 0:
            raise ValueError("Data source must contain at least 1 timestamp.")
        sort_idx = np.argsort(timestamps)
        timestamps = [timestamps[i] for i in sort_idx]
        values = [values[i] for i in sort_idx]
        return timestamps, values

    if isinstance(data_source, pd.DataFrame):
        if time_col not in data_source.columns or data_col not in data_source.columns:
            raise ValueError(f"Missing required columns in DataFrame: {time_col}, {data_col}")
        return sort_by_time(data_source[time_col], data_source[data_col])

    elif isinstance(data_source, str):
        path = Path(data_source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {data_source}")

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            if time_col not in df.columns or data_col not in df.columns:
                raise ValueError(f"Missing required columns in CSV: {time_col}, {data_col}")
            return sort_by_time(df[time_col], df[data_col])

        elif path.suffix.lower() == ".npz":
            npz = np.load(path, allow_pickle=True)
            if time_col not in npz or data_col not in npz:
                raise ValueError(f"Missing keys in NPZ: {time_col}, {data_col}")
            return sort_by_time(npz[time_col], npz[data_col])

        elif path.suffix.lower() == ".json":
            with open(path, "r") as f:
                data_dict = json.load(f)
            if time_col not in data_dict or data_col not in data_dict:
                raise ValueError(f"Missing keys in JSON: {time_col}, {data_col}")
            return sort_by_time(data_dict[time_col], data_dict[data_col])

        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    elif isinstance(data_source, dict):
        if time_col not in data_source or data_col not in data_source:
            raise ValueError(f"Missing keys in dict: {time_col}, {data_col}")
        return sort_by_time(data_source[time_col], data_source[data_col])

    else:
        raise TypeError(f"Unsupported data_source type: {type(data_source).__name__}")


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
            self._duration = self._timestamps[-1] + timestep
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
