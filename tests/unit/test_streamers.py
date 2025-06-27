import os
import numpy as np
import pandas as pd
import json
import pytest
from pathlib import Path
from vidplot.streamers import (
    VideoStreamer, TabularStreamer, StaticStreamer, StaticTabularStreamer, ProgressStreamer
)

test_data_dir = Path("tests/test_data")
video_dir = test_data_dir / "videos"
label_dir = test_data_dir / "frame_labels"

@pytest.mark.parametrize("backend", ["opencv"])  # Add more if available
@pytest.mark.parametrize("video_file", ["moving_square.mp4", "static_test.mp4", "short_test.mp4"])
def test_video_streamer(video_file, backend):
    video_path = video_dir / video_file
    streamer = VideoStreamer(backend, name="vid", path=str(video_path), sample_rate=30.0)
    frames = []
    for t, frame in streamer:
        assert frame is not None
        frames.append(frame)
    assert len(frames) > 0
    assert frames[0].ndim == 3
    assert streamer.size[0] > 0 and streamer.size[1] > 0
    assert streamer.approx_duration > 0

def test_tabular_streamer_csv():
    csv_path = label_dir / "moving_square.csv"
    streamer = TabularStreamer(
        name="tabular_csv",
        data_source=str(csv_path),
        data_col="label",
        time_col="time",
        sample_rate=30.0,
        stream_method="nearest"
    )
    values = []
    for t, v in streamer:
        values.append(v)
    assert len(values) > 0
    assert set(values).issubset({0, 1})
    assert streamer.duration > 0

def test_tabular_streamer_json():
    json_path = label_dir / "moving_square.json"
    streamer = TabularStreamer(
        name="tabular_json",
        data_source=str(json_path),
        data_col="label",
        time_col="time",
        sample_rate=30.0,
        stream_method="nearest_left"
    )
    values = [v for _, v in streamer]
    assert len(values) > 0
    assert set(values).issubset({0, 1})

def test_tabular_streamer_npz():
    npz_path = label_dir / "moving_square.npz"
    streamer = TabularStreamer(
        name="tabular_npz",
        data_source=str(npz_path),
        data_col="label",
        time_col="time",
        sample_rate=30.0,
        stream_method="nearest_right"
    )
    values = [v for _, v in streamer]
    assert len(values) > 0
    assert set(values).issubset({0, 1})

def test_tabular_streamer_dict():
    dict_path = label_dir / "moving_square_dict.json"
    with open(dict_path) as f:
        data = json.load(f)
    streamer = TabularStreamer(
        name="tabular_dict",
        data_source=data,
        data_col="label",
        time_col="time",
        sample_rate=30.0,
        stream_method="nearest"
    )
    values = [v for _, v in streamer]
    assert len(values) > 0
    assert set(values).issubset({0, 1})

def test_static_streamer():
    streamer = StaticStreamer("static", data=42)
    for i, (t, v) in enumerate(streamer):
        assert v == 42
        if i > 10:
            break

def test_static_tabular_streamer():
    csv_path = label_dir / "moving_square.csv"
    streamer = StaticTabularStreamer(
        name="static_tabular",
        data_source=str(csv_path),
        data_col="label",
        time_col="time",
        sample_rate=30.0,
        num_samples=10
    )
    data = streamer._generate_data()
    assert isinstance(data, list)
    assert len(data) == 10

def test_progress_streamer():
    csv_path = label_dir / "moving_square.csv"
    tab_streamer = TabularStreamer(
        name="tabular_csv",
        data_source=str(csv_path),
        data_col="label",
        time_col="time",
        sample_rate=30.0,
    )
    prog = ProgressStreamer("progress", tab_streamer, sample_rate=30.0)
    progresses = [v for _, v in prog]
    assert all(0.0 <= p <= 1.0 for p in progresses)
    assert progresses[-1] <= 1.0 