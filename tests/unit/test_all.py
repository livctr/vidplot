import os
import numpy as np

import pytest

from vidplot.core import AnnotationOrchestrator, StaticDataStreamer
from vidplot.streamers import (
    VideoStreamer,
)
from vidplot.renderers import (
    StringRenderer,
)


def test_annotated_video():
    os.makedirs("tests/output/renderers", exist_ok=True)
    # Use moving_square.mp4 and moving_square.csv for aligned case
    video_path = "tests/input/videos/moving_square.mp4"
    label_path = "tests/input/frame_labels/moving_square.csv"
    # Use short_test.mp4 and moving_square.csv for misaligned case
    video_path_short = "tests/input/videos/short_test.mp4"
    label_path_short = "tests/input/frame_labels/moving_square.csv"

    for video, label, outname in [
        (video_path, label_path, "annotated_video_aligned.mp4"),
        (video_path_short, label_path_short, "annotated_video_misaligned.mp4"),
        (video_path, label_path, "annotated_video_aligned.jpg"),
        (video_path_short, label_path_short, "annotated_video_misaligned.jpg"),
    ]:
        # Video streamer
        vid_streamer = VideoStreamer("video", video, sample_rate=30.0)
        # Label streamer
        label_streamer = StaticTabularStreamer(
            name="labels",
            data_source=label,
            data_col="label",
            time_col="time",
            sample_rate=30.0,
        )
        # Progress streamer
        with pytest.raises(
            ValueError,
            match="Progress streamer must track progress from a streamer of known duration.",
        ):
            progress_streamer = ProgressStreamer("progress", vid_streamer, sample_rate=30.0)

        progress_streamer = ProgressStreamer("progress", label_streamer, sample_rate=30.0)

        # Title streamer (static)
        title = os.path.basename(outname).replace("_", " ").replace(".mp4", "").title()
        title_streamer = StaticDataStreamer("title", title)

        # Compose grid: 3 rows (title, label bar+progress, video), 1 column
        grid_rows = [40, 30, vid_streamer.size[1]]
        grid_cols = [vid_streamer.size[0]]
        orch = AnnotationOrchestrator(grid_rows, grid_cols, gap=0)

        # Renderers
        title_renderer = StringRenderer(
            name="title",
            data_streamer=title_streamer,
            grid_row=(1, 1),
            grid_column=(1, 1),
            font_scale=1.0,
            font_color=(0, 0, 0),
            thickness=2,
            num_expected_lines=1,
        )
        label_bar_renderer = HorizontalLabelBarRenderer(
            name="label_bar",
            data_streamer=label_streamer,
            label_to_color={0: (255, 0, 0), 1: (0, 255, 0)},
            grid_row=(2, 2),
            grid_column=(1, 1),
            height=20,
        )
        progress_renderer = ProgressRenderer(
            name="progress",
            data_streamer=progress_streamer,
            grid_row=(2, 2),
            grid_column=(1, 1),
            bar_color=(0, 0, 255),
            thickness=4,
        )
        # Video renderer (use RGBRenderer if available, else fallback to direct)
        from vidplot.renderers.rgb_renderer import RGBRenderer

        video_renderer = RGBRenderer(
            name="video",
            data_streamer=vid_streamer,
            grid_row=(3, 3),
            grid_column=(1, 1),
        )

        # Set up orchestrator
        orch.set_annotators(
            [vid_streamer, label_streamer, progress_streamer, title_streamer],
            [
                video_renderer,
                label_bar_renderer,
                progress_renderer,
                title_renderer,
            ],
            [
                ("video", "video"),
                ("labels", "label_bar"),
                ("progress", "progress"),
                ("title", "title"),
            ],
        )
        outpath = f"tests/output/all/{outname}"
        orch.write(outpath, fps=30.0)
        assert os.path.exists(outpath)


def test_label_bars():
    os.makedirs("tests/output/all", exist_ok=True)
    # --- Synthetic video data ---
    video_frames = []
    n_frames = 60
    height, width = 64, 384
    for i in range(n_frames):
        # Simple gradient + moving square
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        x = int((width - 20) * i / (n_frames - 1))
        frame[20:40, x:x+20] = [0, 128, 255]
        video_frames.append(frame)
    video_frames = np.stack(video_frames)

    LABEL_TO_COLOR = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}

    # --- Label bar 1 ---
    bar1_labels = [0, 1, 2, 1, 0]
    bar1_timestamps = [0, 10, 25, 45, 55]
    bar1_duration = 60
    bar1_duration_sec = bar1_duration / 30.0

    # --- Label bar 2 ---
    bar2_labels = [1, 2, 0, 2, 1]
    bar2_timestamps = [0, 12, 25, 43, 55]
    bar2_duration = 59
    bar2_duration_sec = bar2_duration / 30.0

    # --- Streamers ---
    from vidplot.core import StaticDataStreamer
    from vidplot.streamers import StaticDataStreamer, LabelBarStreamer, TimestampedDataStreamer
    from vidplot.renderers import StringRenderer, LabelBarRenderer

    # Title
    title = "Label Bar Demo"
    title_streamer = StaticDataStreamer("title", title)
    title_renderer = StringRenderer(
        name="title",
        data_streamer=title_streamer,
        grid_row=(1, 1),
        grid_column=(1, 1),
    )

    # Label bar streamer
    bar1_streamer = LabelBarStreamer(
        "bar1",
        {"label": bar1_labels, "timestamp": bar1_timestamps},
        data_col="label",
        time_col="timestamp",
        duration=bar1_duration,
    )
    bar1_renderer = LabelBarRenderer(
        name="label_bar1",
        data_streamer=bar1_streamer,
        label_to_color=LABEL_TO_COLOR,
        grid_row=(2, 2),
        grid_column=(1, 1),
        z_index=0,
        height=20,
        progress_bar_color=(0, 0, 0),
        write_sampled_data_str=True,
    )

    bar2_streamer = LabelBarStreamer(
        "bar2",
        {"label": bar2_labels, "timestamp": bar2_timestamps},
        data_col="label",
        time_col="timestamp",
        duration=bar1_duration,
    )
    bar2_renderer = LabelBarRenderer(
        name="label_bar2",
        data_streamer=bar2_streamer,
        label_to_color=LABEL_TO_COLOR,
        grid_row=(3, 3),
        grid_column=(1, 1),
        z_index=0,
        height=20,
        progress_bar_color=(0, 0, 0),
        write_sampled_data_str=True,
    )

    grid_rows = [40, 20, 20]
    grid_cols = [width]
    orch = AnnotationOrchestrator(grid_rows, grid_cols, gap=0)
    orch.set_annotators(
        [bar1_streamer, bar2_streamer, title_streamer],
        [bar1_renderer, bar2_renderer, title_renderer],
        [
            ("bar1", "label_bar1"),
            ("bar2", "label_bar2"),
            ("title", "title"),
        ],
    )

    # --- Grid ---
    outpath_mp4 = "tests/output/all/label_bar_demo.mp4"
    orch.write(outpath_mp4, fps=30.0)
    assert os.path.exists(outpath_mp4)


def main():
    # test_annotated_video()
    test_label_bars()


if __name__ == "__main__":
    main()
