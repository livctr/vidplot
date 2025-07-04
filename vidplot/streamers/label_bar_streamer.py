import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from vidplot.core import DataStreamer
from vidplot.streamers.utils import _load_and_validate_data_source


class LabelBarStreamer(DataStreamer):
    """
    A streamer for label bar data that provides uniform sampling of timestamped labels.

    This streamer takes timestamped label data and creates a uniformly sampled stream
    suitable for rendering label bars. It interpolates between label changes to provide
    smooth transitions and consistent sampling rates.

    Examples
    --------

    Using arrays directly:
    >>> times = [0.0, 1.0, 2.0, 3.0]
    >>> labels = ['A', 'B', 'C', 'D']
    >>> streamer = LabelBarStreamer("test", time=times, data=labels, duration=5.0)

    Using pandas DataFrame:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'time': [0, 1, 2], 'label': ['A', 'B', 'C']})
    >>> streamer = LabelBarStreamer("test", data_source=df, time='time', data='label', duration=5.0)

    Using CSV file:
    >>> streamer = LabelBarStreamer("test",
                                    data_source="labels.csv",
                                    time='time',
                                    data='label',
                                    duration=5.0)

    Using dictionary:
    >>> data_dict = {'time': [0, 1, 2], 'label': ['A', 'B', 'C']}
    >>> streamer = LabelBarStreamer("test",
                                    data_source=data_dict,
                                    time='time',
                                    data='label',
                                    duration=5.0)

    Parameters
    ----------
    name : str
        Unique name for the streamer.
    data_source : DataFrame, str, or dict, optional
        Data source. If DataFrame, time and data should be column names.
        If str, should be file path (CSV, NPZ, JSON). If dict, time and data should be keys.
    time : str or array-like
        Time/timestamp data. If str, column name or key in data_source.
        If array-like, actual timestamp values.
    data : str or array-like
        Label data. If str, column name or key in data_source.
        If array-like, actual label values.
    duration : float, optional
        Total duration in seconds. If None, uses the last timestamp.
    num_samples : int, default=1000
        Number of uniform samples to generate over the duration.
    round_decimals : int, default=3
        Number of decimal places to round timestamps.

    Notes
    -----
    - If both time and data are strings, they are treated as column names/keys in data_source.
    - If both time and data are iterables, they are treated as direct data arrays.
    - Data is automatically sorted by timestamp.
    - Supported file formats: CSV, NPZ, JSON.
    - When using arrays directly, time and data must have the same length.
    - The streamer creates uniform samples over the specified duration.
    - Each iteration returns (timestamp, (all_labels, current_label, progress)).
    """

    def __init__(
        self,
        name: str,
        time: Union[str, Iterable],
        data: Union[str, Iterable],
        data_source: Optional[Union[pd.DataFrame, str, Dict[str, Iterable]]] = None,
        duration: Optional[float] = None,
        num_samples: int = 1000,
        round_decimals: int = 3,
    ) -> None:
        super().__init__(name=name)

        # Check if both time and data are strings (column names/keys)
        if isinstance(time, str) and isinstance(data, str):
            if data_source is None:
                raise ValueError("data_source must be provided when time and data are strings")
            self._timestamps, self._data = _load_and_validate_data_source(data_source, data, time)
        # Check if both time and data are iterables (direct data)
        elif hasattr(time, "__iter__") and hasattr(data, "__iter__"):
            # Create a dict and feed to _load_and_validate_data_source
            data_dict = {"time": time, "data": data}
            self._timestamps, self._data = _load_and_validate_data_source(data_dict, "data", "time")
        else:
            raise ValueError(
                "Both time and data must be either strings (column names) or iterables "
                "(direct data). If strings, provide data_source. If iterables, provide "
                "the actual data arrays."
            )

        self._timestamps = np.asarray(self._timestamps, dtype=float)
        self._timestamps = np.round(self._timestamps, round_decimals)
        if len(self._timestamps) == 0:
            raise ValueError("No data found in the given source")

        # determine total duration
        last_ts = float(self._timestamps[-1])
        self._duration = float(duration) if duration is not None else last_ts
        assert self._duration > 0, "Duration must be positive"

        # internal pointer & flag for extra emit
        self._idx = 0
        self._emitted_extra = False

        ts_uniform = np.linspace(0.0, self._duration, num_samples, endpoint=True)
        ts_uniform = np.round(ts_uniform, round_decimals)
        data_out = []
        for t in ts_uniform:
            i = np.searchsorted(self._timestamps, t, side="right") - 1
            idx = max(0, int(i))
            data_out.append(self._data[idx])
        self._ts_uniform = ts_uniform.tolist()  # len == num_samples
        self._data_sampled = data_out  # len == num_samples
        self._idx = 0

    @property
    def duration(self) -> float:
        """Total duration of the stream in seconds."""
        return self._duration

    def __next__(self) -> Tuple[float, Any]:
        if self._idx >= len(self._ts_uniform):
            raise StopIteration

        ts = float(self._ts_uniform[self._idx])
        sampled_value = self._data_sampled[self._idx]
        norm = ts / self._duration

        self._idx += 1
        return ts, (self._data_sampled, sampled_value, norm)
