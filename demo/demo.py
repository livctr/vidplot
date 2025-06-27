import os


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

from vidplot.renderers.rgb_renderer import RGBRenderer


if __name__ == "__main__":

    os.makedirs("demo/output/", exist_ok=True)
    video_path = "demo/assets/moving_square.mp4"
    label_path = "demo/assets/moving_square.csv"
    outnames = [
        "annotated_video.png",  # Displays just the first frame
        "annotated_video.mp4",
    ]

    for outname in outnames:

        # Video streamer
        vid_streamer = VideoStreamer("opencv", name="video", path=video_path, sample_rate=30.0)

        # Static tabular streamer, for displaying the blue/orange bar at the top
        label_streamer = StaticTabularStreamer(
            name="labels",
            data_source=label_path,
            data_col="label",
            time_col="time",
            sample_rate=30.0,
        )

        # Progress streamer, for displaying the red vertical bar that moves
        # along the blue/orange bar
        progress_streamer = ProgressStreamer("progress", label_streamer, sample_rate=30.0)

        # Title streamer (static)
        title = os.path.basename(outname).replace("_", " ").replace(".mp4", "").title()
        title_streamer = StaticDataStreamer("title", title)

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

        video_renderer = RGBRenderer(
            name="video",
            data_streamer=vid_streamer,
            grid_row=(3, 3),
            grid_column=(1, 1),
        )

        # Set up orchestrator

        # Compose grid: 3 rows (title, label bar+progress, video), 1 column
        grid_rows = [40, 30, vid_streamer.size[1]]
        grid_cols = [vid_streamer.size[0]]
        orch = AnnotationOrchestrator(grid_rows, grid_cols, gap=0)

        # Set the annotators
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

        # Run
        outpath = "demo/output/" + outname
        os.makedirs("demo/output/", exist_ok=True)
        orch.write(outpath, fps=30.0)
