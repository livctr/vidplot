import json
import pytest
from pathlib import Path
from vidplot.core import StaticDataStreamer
from vidplot.streamers import (
    VideoStreamer,
    TabularStreamer,
    StaticTabularStreamer,
    ProgressStreamer,
)

input_dir = Path("tests/input")
video_dir = input_dir / "videos"
label_dir = input_dir / "frame_labels"


@pytest.mark.parametrize("backend", ["opencv", "pyav", "decord"])
@pytest.mark.parametrize("video_file", ["moving_square.mp4", "static_test.mp4", "short_test.mp4"])
def test_video_streamer(video_file, backend):
    video_path = video_dir / video_file
    streamer = VideoStreamer("vid", str(video_path), backend, sample_rate=30.0)

    if video_file == "moving_square.mp4":
        expected_size = (640, 480)
        expected_duration = 5.0
    elif video_file == "static_test.mp4":
        expected_size = (320, 240)
        expected_duration = 3.0
    elif video_file == "short_test.mp4":
        expected_size = (320, 240)
        expected_duration = 2.0
    else:
        raise ValueError(f"Unknown test video: {video_file}")
    assert streamer.size == expected_size  # (W, H)
    assert abs(streamer.approx_duration - expected_duration) <= 1e-5

    cnt = 0
    for t, frame in streamer:
        assert frame is not None
        assert frame.shape == (
            expected_size[1],
            expected_size[0],
            3,
        )  # (H, W, 3)
        print(t, cnt)
        cnt += 1
    assert cnt > 0


def test_tabular_streamer():

    # Test 4 different loading methods
    dict_path = label_dir / "moving_square_dict.json"
    with open(dict_path) as f:
        data = json.load(f)
    data_sources = [
        str(label_dir / "moving_square.csv"),
        str(label_dir / "moving_square.json"),
        str(label_dir / "moving_square.npz"),
        data,
    ]
    methods = ["csv", "json", "npz", "dictionary"]
    streamers = [
        TabularStreamer(
            name="tabular_csv",
            data_source=data_source,
            data_col="label",
            time_col="time",
            sample_rate=30.0,
            stream_method="nearest",
        )
        for data_source in data_sources
    ]

    for method, streamer in zip(methods, streamers):
        values = []
        for t, v in streamer:
            values.append(v)
        assert len(values) > 0, f"Error with loading method: {method}"
        assert set(values).issubset({0, 1}), f"Error with loading method: {method}"
        assert streamer.duration > 0, f"Error with loading method: {method}"

        # Run-length encode the output
        rle = []
        prev = values[0]
        count = 1
        for val in values[1:]:
            if val == prev:
                count += 1
            else:
                rle.append((prev, count))
                prev = val
                count = 1

        rle.append((prev, count))

        # One second off then one second on
        for i in range(len(rle)):
            assert rle[i][0] == (i % 2), f"Error with loading method: {method}"

        # Each run should be about 30 (1 second at 30 fps), allow off by 1
        for label, runlen in rle:
            assert (
                abs(runlen - 30) <= 1
            ), f"Error with loading method: {method}. Run of {label}'s is {runlen}, expected ~30"


def test_static_streamer():
    streamer = StaticDataStreamer("static", data=42)
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
        num_samples=10,
    )

    for i, (t, v) in enumerate(streamer):
        assert isinstance(v, list)
        assert len(v) == 10

        # 10 samples over 5 seconds. Expect 0, 0, 1, 1, 0, 0, 1, 1,0, 0
        for j in range(10):
            assert abs(v[j] - ((j // 2) % 2)) <= 1e-5

        if i > 10:
            break


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
    assert all(0.0 - 1e-5 <= p <= 1.0 + 1e-5 for p in progresses)
    assert abs(progresses[0] - 0.0) <= 1e-5
    assert abs(progresses[-1] - 1.0) <= 1e-5
