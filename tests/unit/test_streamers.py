import json
import pytest
from pathlib import Path
from vidplot.streamers import (
    VideoStreamer,
    TimestampedDataStreamer,
    StaticDataStreamer,
)

input_dir = Path("tests/input")
video_dir = input_dir / "videos"
label_dir = input_dir / "frame_labels"


@pytest.mark.parametrize("backend", ["opencv", "pyav", "decord"])
@pytest.mark.parametrize("video_file", ["moving_square.mp4", "static_test.mp4", "short_test.mp4"])
def test_video_streamer(video_file, backend):
    video_path = video_dir / video_file
    streamer = VideoStreamer("vid", str(video_path), backend)

    if video_file == "moving_square.mp4":
        expected_size = (480, 640)
        expected_duration = 5.0
    elif video_file == "static_test.mp4":
        expected_size = (240, 320)
        expected_duration = 3.0
    elif video_file == "short_test.mp4":
        expected_size = (240, 320)
        expected_duration = 2.0
    else:
        raise ValueError(f"Unknown test video: {video_file}")
    assert streamer.size == expected_size  # (W, H)
    assert abs(streamer.duration - expected_duration) <= 1e-5

    cnt = 0
    for t, frame in streamer:
        assert frame is not None
        assert frame.shape == (
            expected_size[0],
            expected_size[1],
            3,
        )  # (H, W, 3)
        print(t, cnt)
        cnt += 1
    assert cnt > 0


def test_timestamped_streamer_legacy():
    # Test legacy API (data_source, data_col, time_col)
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
        TimestampedDataStreamer(
            name="tabular_csv",
            data_source=data_source,
            time="time",
            data="label",
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


def test_timestamped_streamer_new_api():
    # Test new unified API (time, data)
    import pandas as pd

    times = [0, 1, 2, 3, 4, 5]
    labels = [0, 0, 1, 1, 0, 0]
    df = pd.DataFrame({"time": times, "label": labels})
    streamer = TimestampedDataStreamer(
        name="new_api",
        data_source=df,
        time="time",
        data="label",
    )
    out_times = []
    out_labels = []
    for t, v in streamer:
        out_times.append(t)
        out_labels.append(v)
    assert out_times == [0, 1, 2, 3, 4, 5]
    assert out_labels == labels


def test_timestamped_streamer_direct_arrays():
    # Test direct array usage
    times = [0, 1, 2, 3, 4, 5]
    labels = [0, 0, 1, 1, 0, 0]
    streamer = TimestampedDataStreamer(
        name="direct_arrays",
        time=times,
        data=labels,
    )
    out_times = []
    out_labels = []
    for t, v in streamer:
        out_times.append(t)
        out_labels.append(v)
    assert out_times == [0, 1, 2, 3, 4, 5]
    assert out_labels == labels


def test_static_streamer():
    streamer = StaticDataStreamer("static", data=42)
    for i, (t, v) in enumerate(streamer):
        assert v == 42
        if i > 10:
            break
