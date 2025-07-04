import numpy as np
import pandas as pd
import json
from typing import Any, Dict, Iterable, List, Tuple, Union
from pathlib import Path


def _load_and_validate_data_source(
    data_source: Union[pd.DataFrame, str, Dict[str, Any]],
    data_col: str,
    time_col: str,
) -> Tuple[List[float], List[Any]]:
    """
    Loads time-series data from various formats and validates required columns.

    This is the legacy function that uses data_col and time_col parameters.
    For the new seaborn-like API, use _load_seaborn_like_data instead.
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


def _load_seaborn_like_data(
    data: Union[pd.DataFrame, str, Dict[str, Any]], x: Union[str, Iterable], y: Union[str, Iterable]
) -> Tuple[List[float], List[Any]]:
    """
    Loads time-series data using seaborn-like API with data, x, and y parameters.

    Parameters
    ----------
    data : DataFrame, str, or dict
        Data source. If DataFrame, x and y should be column names.
        If str, should be file path (CSV, NPZ, JSON).
        If dict, x and y should be keys.
    x : str or array-like
        Time/timestamp data. If str, column name or key.
        If array-like, actual timestamp values.
    y : str or array-like
        Data values. If str, column name or key.
        If array-like, actual data values.

    Returns
    -------
    tuple
        (timestamps, data_values) as lists
    """
    # Case 1: x and y are array-like (numpy arrays, lists, etc.)
    if not isinstance(x, str) and not isinstance(y, str):
        # Direct arrays provided
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y)

        if len(x_array) != len(y_array):
            raise ValueError(
                f"Length mismatch: x has {len(x_array)} elements, y has {len(y_array)} elements"
            )

        # Sort by time
        sort_idx = np.argsort(x_array)
        timestamps = x_array[sort_idx].tolist()
        data_values = [y_array[i] for i in sort_idx]

        return timestamps, data_values

    # Case 2: x and y are column names/keys, data is DataFrame/dict/file
    if isinstance(x, str) and isinstance(y, str):
        # Use the existing utility function with x as time_col and y as data_col
        return _load_and_validate_data_source(data, y, x)

    # Case 3: Mixed case - one is string, one is array
    raise ValueError(
        "x and y must both be either strings (column names/keys) or array-like objects. "
        f"Got x={type(x).__name__}, y={type(y).__name__}"
    )
