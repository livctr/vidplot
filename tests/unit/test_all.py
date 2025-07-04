import os
import numpy as np


from vidplot.core import VideoCanvas, StaticDataStreamer
from vidplot.streamers import VideoStreamer, LabelBarStreamer
from vidplot.renderers import (
    StringRenderer,
    LabelBarRenderer,
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
        vid_streamer = VideoStreamer("video", video)
        # Label streamer (use TimestampedDataStreamer)

        label_streamer = LabelBarStreamer("labels", "time", "label", label)
        # Title streamer (static)
        title = os.path.basename(outname).replace("_", " ").replace(".mp4", "").title()
        title_streamer = StaticDataStreamer("title", title)
        # Compose grid: 3 rows (title, label bar, video), 1 column
        canvas = VideoCanvas(row_gap=0, col_gap=0)
        # Renderers
        title_renderer = StringRenderer(
            name="title",
            data_streamer=title_streamer,
        )
        label_bar_renderer = LabelBarRenderer(
            name="label_bar",
            data_streamer=label_streamer,
            label_to_color={0: (255, 0, 0), 1: (0, 255, 0)},
            height=20,
        )
        from vidplot.renderers.rgb_renderer import RGBRenderer

        video_renderer = RGBRenderer(
            name="video",
            data_streamer=vid_streamer,
        )
        # Attach streamers/renderers to canvas
        video_h, video_w = vid_streamer.size[:2]
        canvas.attach(
            title_streamer,
            title_renderer,
            grid_row=1,
            grid_col=1,
            height=[40],
            width=[video_w],
            z_index=0,
        )
        canvas.attach(
            label_streamer,
            label_bar_renderer,
            grid_row=2,
            grid_col=1,
            height=[30],
            width=[video_w],
            z_index=0,
        )
        canvas.attach(
            vid_streamer,
            video_renderer,
            grid_row=3,
            grid_col=1,
            height=[video_h],
            width=[video_w],
            z_index=0,
        )
        outpath = f"tests/output/all/{outname}"
        canvas.write(outpath, fps=30.0)
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
        frame[20:40, x : x + 20] = [0, 128, 255]
        video_frames.append(frame)
    video_frames = np.stack(video_frames)
    LABEL_TO_COLOR = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
    # --- Label bar 1 ---
    bar1_labels = [0, 1, 2, 1, 0]
    bar1_timestamps = [0, 10, 25, 45, 55]
    bar1_duration = 60
    # --- Label bar 2 ---
    bar2_labels = [1, 2, 0, 2, 1]
    bar2_timestamps = [0, 12, 25, 43, 55]
    # bar2_duration = 59
    from vidplot.streamers import StaticDataStreamer, LabelBarStreamer
    from vidplot.renderers import StringRenderer, LabelBarRenderer

    # Title
    title = "Label Bar Demo"
    title_streamer = StaticDataStreamer("title", title)
    title_renderer = StringRenderer(
        name="title",
        data_streamer=title_streamer,
    )
    # Label bar streamer
    bar1_streamer = LabelBarStreamer(
        name="bar1", data=bar1_labels, time=bar1_timestamps, duration=bar1_duration
    )
    bar1_renderer = LabelBarRenderer(
        name="label_bar1",
        data_streamer=bar1_streamer,
        label_to_color=LABEL_TO_COLOR,
        height=20,
        progress_bar_color=(0, 0, 0),
        write_sampled_data_str=True,
    )
    bar2_streamer = LabelBarStreamer("bar2", data=bar2_labels, time=bar2_timestamps)
    bar2_renderer = LabelBarRenderer(
        name="label_bar2",
        data_streamer=bar2_streamer,
        label_to_color=LABEL_TO_COLOR,
        height=20,
        progress_bar_color=(0, 0, 0),
        write_sampled_data_str=True,
    )
    # Use VideoCanvas and attach API
    canvas = VideoCanvas(row_gap=0, col_gap=0)
    # Row 1: title (height 40), Row 2: bar1 (height 20), Row 3: bar2 (height 20)
    # Col 1: only one column (width = width)
    canvas.attach(
        title_streamer,
        title_renderer,
        grid_row=1,
        grid_col=1,
        height=[40],
        width=[width],
        z_index=0,
    )
    canvas.attach(
        bar1_streamer, bar1_renderer, grid_row=2, grid_col=1, height=[20], width=[width], z_index=0
    )
    canvas.attach(
        bar2_streamer, bar2_renderer, grid_row=3, grid_col=1, height=[20], width=[width], z_index=0
    )
    # --- Grid ---
    outpath_mp4 = "tests/output/all/label_bar_demo.png"
    canvas.write(outpath_mp4, fps=30.0)
    assert os.path.exists(outpath_mp4)


def main():
    test_annotated_video()
    test_label_bars()


if __name__ == "__main__":
    main()
