import os

import pytest

from vidplot.core import AnnotationOrchestrator, StaticDataStreamer
from vidplot.streamers import (
    VideoStreamer,
    ProgressStreamer,
    StaticTabularStreamer,
)
from vidplot.renderers import (
    StringRenderer,
    HorizontalLabelBarRenderer,
    ProgressRenderer,
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
        vid_streamer = VideoStreamer("opencv", name="video", path=video, sample_rate=30.0)
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


def main():
    test_annotated_video()


if __name__ == "__main__":
    main()
