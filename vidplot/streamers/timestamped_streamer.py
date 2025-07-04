import pandas as pd
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from vidplot.core import DataStreamer
from vidplot.streamers.utils import _load_and_validate_data_source


class TimestampedDataStreamer(DataStreamer):
    """
    A tabular data streamer that emits timestamped data points in chronological order.

    This streamer:
    1) Emits each original (timestamp, payload) in order, and then
    2) If `duration` > last timestamp, emits exactly one more (duration, last_payload)
        before stopping.

    If `duration` is omitted or ≤ last timestamp, only the raw data is emitted.

    Examples
    --------

    Using arrays directly:
    >>> times = [0.0, 1.0, 2.0, 3.0]
    >>> values = ['A', 'B', 'C', 'D']
    >>> streamer = TimestampedDataStreamer("test", time=times, data=values)

    Using pandas DataFrame:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'time': [0, 1, 2], 'label': ['A', 'B', 'C']})
    >>> streamer = TimestampedDataStreamer("test", data_source=df, time='time', data='label')

    Using CSV file:
    >>> streamer = TimestampedDataStreamer("test",
                                           data_source="data.csv",
                                           time='time',
                                           data='label')

    Using dictionary:
    >>> data_dict = {'time': [0, 1, 2], 'label': ['A', 'B', 'C']}
    >>> streamer = TimestampedDataStreamer("test", data_source=data_dict, time='time', data='label')

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
        Data values. If str, column name or key in data_source.
        If array-like, actual data values.
    duration : float, optional
        Total duration in seconds. If None, uses the last timestamp.

    Notes
    -----
    - If both time and data are strings, they are treated as column names/keys in data_source.
    - If both time and data are iterables, they are treated as direct data arrays.
    - Data is automatically sorted by timestamp.
    - Supported file formats: CSV, NPZ, JSON.
    - When using arrays directly, time and data must have the same length.
    """

    def __init__(
        self,
        name: str,
        time: Union[str, Iterable],
        data: Union[str, Iterable],
        data_source: Optional[Union[pd.DataFrame, str, Dict[str, Iterable]]] = None,
        duration: Optional[float] = None,
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
                "Both time and data must be either strings (column names) or iterables"
                "(direct data). If strings, provide data_source. If iterables, provide"
                " the actual data arrays."
            )

        if not self._timestamps:
            raise ValueError("No data found in the given source")

        # determine total duration
        last_ts = float(self._timestamps[-1])
        self._duration = float(duration) if duration is not None else last_ts

        # internal pointer & flag for extra emit
        self._idx = 0
        self._emitted_extra = False

    @property
    def duration(self) -> float:
        """Total duration of the stream in seconds."""
        return self._duration

    def __next__(self) -> Tuple[float, Any]:
        # 1) emit raw data
        if self._idx < len(self._timestamps):
            ts = float(self._timestamps[self._idx])
            payload = self._data[self._idx]
            self._idx += 1
            return ts, payload

        # 2) emit one extra (duration, last_payload) if requested
        last_ts = float(self._timestamps[-1])
        if (not self._emitted_extra) and (self._duration > last_ts):
            self._emitted_extra = True
            return self._duration, self._data[-1]

        # 3) done
        raise StopIteration
