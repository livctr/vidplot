import os
from vidplot.streamers import VideoStreamer, LabelBarStreamer, StaticDataStreamer, TimestampedDataStreamer
from vidplot.renderers import RGBRenderer, LabelBarRenderer, StringRenderer, BoxRenderer
from vidplot.core.video_canvas import VideoCanvas
from vidplot.style import rcParams, use_style


if __name__ == "__main__":
    # --- DEMONSTRATE GLOBAL STYLE PRESETS ---
    # You can quickly apply a global style preset:
    use_style("minimal")  # Try "default", "dark", "minimal", or "high_contrast"

    # --- DEMONSTRATE CUSTOMIZING GLOBAL STYLES ---
    # You can further customize style parameters after use_style
    rcParams().font_family = "DejaVu Sans"
    rcParams().font_size = 24
    rcParams().font_color = (0, 0, 128)  # Navy blue
    rcParams().string_thickness = 3
    rcParams().label_bar_height = 24
    # You can also use: use_style({"font.size": 28, ...})

    print(
        "\n[VidPlot Demo]\n---\nThis demo shows how to customize global "
        "and per-renderer styles.\nFor more, see vidplot/style/__init__.py"
        " or the documentation.\n---\n"
    )

    os.makedirs("demo/output/", exist_ok=True)
    video_path = "demo/assets/demo_video.mp4"
    label_path = "demo/assets/demo_labels.csv"
    bboxes_path = "demo/assets/demo_bboxes.json"
    outnames = [
        "annotated_video.png",  # Displays just the first frame
        "annotated_video.mp4",
    ]

    for outname in outnames:


        canvas = VideoCanvas(row_gap=5, col_gap=0)

        vid_streamer = VideoStreamer("video", video_path)
        vid_renderer = RGBRenderer("video", vid_streamer)
        height = vid_streamer.size[0]
        width = vid_streamer.size[1]
        canvas.attach(vid_streamer, vid_renderer, grid_row=3, grid_col=1, height=[height], width=[width])

        title_streamer = StaticDataStreamer("title", "Sea Turtle Gliding Over Coral Reef")
        title_renderer = StringRenderer(
            name="title",
            data_streamer=title_streamer,

        )
        canvas.attach(title_streamer, title_renderer, grid_row=1, grid_col=1, height=[20], width=[width])

        label_streamer = LabelBarStreamer(
            name="label_bar",
            data_source=label_path,
            time="time",
            data="label",
            duration=vid_streamer.duration
        )
        label_renderer = LabelBarRenderer(
            "label_bar",
            label_streamer,
            label_to_color={"recover": (86, 113, 121), "propel": (194, 178, 128)}
        )
        canvas.attach(label_streamer, label_renderer, grid_row=2, grid_col=1, width=[width])

        box_streamer = TimestampedDataStreamer("box", "time", "bboxes", data_source=bboxes_path, duration=vid_streamer.duration)
        box_renderer = BoxRenderer(
            name="box",
            data_streamer=box_streamer,
            id_to_color={1: (86, 113, 121)},
            box_color=(86, 113, 121),
            thickness=2,
            box_representation_format="xyxy"
        )
        canvas.attach(box_streamer, box_renderer, grid_row=3, grid_col=1, z_index=1)

        canvas.show_layout("demo/output/" + "layout.png")
        outpath = "demo/output/" + outname
        os.makedirs("demo/output/", exist_ok=True)
        canvas.write(outpath, fps=30.0)

# For more on styling, see vidplot/style/__init__.py or the documentation.
