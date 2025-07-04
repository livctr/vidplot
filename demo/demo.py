import os
from vidplot.streamers import VideoStreamer, LabelBarStreamer, StaticDataStreamer
from vidplot.renderers import RGBRenderer, LabelBarRenderer, StringRenderer
from vidplot.core.video_canvas import VideoCanvas
from vidplot.style import rcParams, use_style


if __name__ == "__main__":
    # --- DEMONSTRATE GLOBAL STYLE PRESETS ---
    # You can quickly apply a global style preset:
    use_style("dark")  # Try "default", "dark", "minimal", or "high_contrast"

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
    video_path = "demo/assets/moving_square.mp4"
    label_path = "demo/assets/moving_square.csv"
    outnames = [
        "annotated_video.png",  # Displays just the first frame
        "annotated_video.mp4",
    ]

    for outname in outnames:
        # Video streamer
        vid_streamer = VideoStreamer("video", video_path)

        # Label bar streamer (using new API)
        label_streamer = LabelBarStreamer(
            name="labels",
            data_source=label_path,
            time="time",
            data="label",
            duration=vid_streamer.duration,
            num_samples=int(vid_streamer.duration * 30),
        )

        # Video renderer (inherits global style)
        video_renderer = RGBRenderer(
            name="video",
            data_streamer=vid_streamer,
        )

        # Label bar renderer (override height and font size locally)
        label_bar_renderer = LabelBarRenderer(
            name="label_bar",
            data_streamer=label_streamer,
            height=32,  # Override global label_bar.height
            font_size=18,  # Override global font.size for this renderer
        )

        # String overlay renderer (override font and color locally)
        overlay_text = os.path.basename(outname).replace("_", " ").replace(".mp4", "").title()
        string_streamer = StaticDataStreamer("string_overlay_streamer", "Overlay!")
        string_renderer = StringRenderer(
            name="string_overlay",
            data_streamer=string_streamer,
            text=overlay_text,
            font_scale=1.2,  # Override global font scale
            font_color=(255, 0, 0),  # Override global font color (red)
            thickness=4,  # Override global thickness
            num_expected_lines=1,
        )

        # Set up VideoCanvas
        height = vid_streamer.size[1]
        width = vid_streamer.size[0]
        canvas = VideoCanvas(row_gap=5, col_gap=0)
        # Attach video to (2,1), label bar to (1,1), string overlay to (2,1) with higher z_index
        canvas.attach(
            vid_streamer,
            video_renderer,
            grid_row=2,
            grid_col=1,
            height=[height],
            width=[width],
            z_index=0,
        )
        canvas.attach(
            label_streamer,
            label_bar_renderer,
            grid_row=1,
            grid_col=1,
            height=[32],
            width=[width],
            z_index=0,
        )
        canvas.attach(
            string_streamer,
            string_renderer,
            grid_row=2,
            grid_col=1,
            height=[height],
            width=[width],
            z_index=1,
        )

        # Show layout (optional)
        canvas.show_layout("demo/output/" + "layout.png")
        # Run
        outpath = "demo/output/" + outname
        os.makedirs("demo/output/", exist_ok=True)
        canvas.write(outpath, fps=30.0)

# For more on styling, see vidplot/style/__init__.py or the documentation.
