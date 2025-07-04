#!/usr/bin/env python3
"""
Demonstration of vidplot's styling system.

This script shows how to use the new styling configuration system
similar to matplotlib's rcParams.
"""

import os
import numpy as np
from vidplot import (
    AnnotationOrchestrator,
    StaticDataStreamer,
    rcParams,
    rc_context,
    use_style,
    set_font_size,
    set_line_width,
    set_marker_size,
    set_color_scheme,
)
from vidplot.streamers import VideoStreamer, ProgressStreamer, StaticTabularStreamer
from vidplot.renderers import (
    StringRenderer,
    HorizontalLabelBarRenderer,
    ProgressRenderer,
    RGBRenderer,
)


def create_demo_video():
    """Create a simple demo video if it doesn't exist."""
    video_path = "demo/assets/style_demo.mp4"
    if not os.path.exists(video_path):
        os.makedirs("demo/assets", exist_ok=True)

        # Create a simple video with moving shapes
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))

        for i in range(90):  # 3 seconds at 30fps
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

            # Moving circle
            x = int(320 + 200 * np.cos(i * 0.1))
            y = int(240 + 150 * np.sin(i * 0.15))
            cv2.circle(frame, (x, y), 30, (0, 0, 255), -1)

            # Moving rectangle
            rect_x = int(100 + 50 * np.sin(i * 0.2))
            rect_y = int(100 + 30 * np.cos(i * 0.25))
            cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 40, rect_y + 40), (0, 255, 0), -1)

            writer.write(frame)

        writer.release()
        print(f"Created demo video: {video_path}")

    return video_path


def create_demo_labels():
    """Create demo label data."""
    import pandas as pd

    label_path = "demo/assets/style_demo.csv"
    if not os.path.exists(label_path):
        os.makedirs("demo/assets", exist_ok=True)

        # Create label data
        times = np.linspace(0, 3, 90)  # 3 seconds, 30fps
        labels = []
        for t in times:
            if t < 1:
                labels.append("Phase 1")
            elif t < 2:
                labels.append("Phase 2")
            else:
                labels.append("Phase 3")

        df = pd.DataFrame({"time": times, "label": labels})
        df.to_csv(label_path, index=False)
        print(f"Created demo labels: {label_path}")

    return label_path


def demo_default_style():
    """Demonstrate default styling."""
    print("\n=== Default Style Demo ===")

    video_path = create_demo_video()
    label_path = create_demo_labels()

    # Set up streamers
    vid_streamer = VideoStreamer("video", video_path, sample_rate=30.0)
    label_streamer = StaticTabularStreamer(
        name="labels",
        data_source=label_path,
        data_col="label",
        time_col="time",
        sample_rate=30.0,
    )
    progress_streamer = ProgressStreamer("progress", label_streamer, sample_rate=30.0)
    title_streamer = StaticDataStreamer("title", "Default Style Demo")

    # Set up renderers with default styling
    title_renderer = StringRenderer(
        name="title",
        data_streamer=title_streamer,
        grid_row=(1, 1),
        grid_column=(1, 1),
    )

    label_bar_renderer = HorizontalLabelBarRenderer(
        name="label_bar",
        data_streamer=label_streamer,
        grid_row=(2, 2),
        grid_column=(1, 1),
    )

    progress_renderer = ProgressRenderer(
        name="progress",
        data_streamer=progress_streamer,
        grid_row=(2, 2),
        grid_column=(1, 1),
    )

    video_renderer = RGBRenderer(
        name="video",
        data_streamer=vid_streamer,
        grid_row=(3, 3),
        grid_column=(1, 1),
    )

    # Set up orchestrator
    grid_rows = [40, 30, vid_streamer.size[1]]
    grid_cols = [vid_streamer.size[0]]
    orch = AnnotationOrchestrator(grid_rows, grid_cols, gap=0)

    orch.set_annotators(
        [vid_streamer, label_streamer, progress_streamer, title_streamer],
        [video_renderer, label_bar_renderer, progress_renderer, title_renderer],
        [
            ("video", "video"),
            ("labels", "label_bar"),
            ("progress", "progress"),
            ("title", "title"),
        ],
    )

    # Render
    os.makedirs("demo/output", exist_ok=True)
    outpath = "demo/output/default_style.mp4"
    orch.write(outpath, fps=30.0)
    print(f"Rendered: {outpath}")


def demo_custom_style():
    """Demonstrate custom styling using rcParams."""
    print("\n=== Custom Style Demo ===")

    # Apply custom styling
    use_style("dark")  # Start with dark theme
    set_font_size(16)  # Larger fonts
    set_line_width(4)  # Thicker lines
    set_marker_size(6)  # Larger markers

    # Custom color scheme
    custom_colors = [
        (255, 100, 100),  # Light red
        (100, 255, 100),  # Light green
        (100, 100, 255),  # Light blue
        (255, 255, 100),  # Light yellow
        (255, 100, 255),  # Light magenta
    ]
    set_color_scheme(custom_colors)

    # Show current configuration
    print("Current style configuration:")
    config = rcParams()
    for key, value in config.to_dict().items():
        if key in ["font_size", "line_width", "marker_size", "label_bar_colors"]:
            print(f"  {key}: {value}")

    video_path = create_demo_video()
    label_path = create_demo_labels()

    # Set up streamers
    vid_streamer = VideoStreamer("video", video_path, sample_rate=30.0)
    label_streamer = StaticTabularStreamer(
        name="labels",
        data_source=label_path,
        data_col="label",
        time_col="time",
        sample_rate=30.0,
    )
    progress_streamer = ProgressStreamer("progress", label_streamer, sample_rate=30.0)
    title_streamer = StaticDataStreamer("title", "Custom Style Demo")

    # Set up renderers (will use global style settings)
    title_renderer = StringRenderer(
        name="title",
        data_streamer=title_streamer,
        grid_row=(1, 1),
        grid_column=(1, 1),
    )

    label_bar_renderer = HorizontalLabelBarRenderer(
        name="label_bar",
        data_streamer=label_streamer,
        grid_row=(2, 2),
        grid_column=(1, 1),
    )

    progress_renderer = ProgressRenderer(
        name="progress",
        data_streamer=progress_streamer,
        grid_row=(2, 2),
        grid_column=(1, 1),
    )

    video_renderer = RGBRenderer(
        name="video",
        data_streamer=vid_streamer,
        grid_row=(3, 3),
        grid_column=(1, 1),
    )

    # Set up orchestrator
    grid_rows = [40, 30, vid_streamer.size[1]]
    grid_cols = [vid_streamer.size[0]]
    orch = AnnotationOrchestrator(grid_rows, grid_cols, gap=0)

    orch.set_annotators(
        [vid_streamer, label_streamer, progress_streamer, title_streamer],
        [video_renderer, label_bar_renderer, progress_renderer, title_renderer],
        [
            ("video", "video"),
            ("labels", "label_bar"),
            ("progress", "progress"),
            ("title", "title"),
        ],
    )

    # Render
    os.makedirs("demo/output", exist_ok=True)
    outpath = "demo/output/custom_style.mp4"
    orch.write(outpath, fps=30.0)
    print(f"Rendered: {outpath}")


def demo_context_manager():
    """Demonstrate using rc_context for temporary styling."""
    print("\n=== Context Manager Demo ===")

    # Reset to default style
    use_style("default")

    video_path = create_demo_video()
    label_path = create_demo_labels()

    # Set up streamers
    vid_streamer = VideoStreamer("video", video_path, sample_rate=30.0)
    label_streamer = StaticTabularStreamer(
        name="labels",
        data_source=label_path,
        data_col="label",
        time_col="time",
        sample_rate=30.0,
    )
    progress_streamer = ProgressStreamer("progress", label_streamer, sample_rate=30.0)
    title_streamer = StaticDataStreamer("title", "Context Manager Demo")

    # Use context manager for temporary styling
    with rc_context(
        {
            "font_scale": 0.8,
            "line_thickness": 3,
            "marker_radius": 5,
            "background_color": (240, 240, 240),
            "font_color": (50, 50, 50),
        }
    ):
        print("Applied temporary styling with rc_context")

        # Set up renderers
        title_renderer = StringRenderer(
            name="title",
            data_streamer=title_streamer,
            grid_row=(1, 1),
            grid_column=(1, 1),
        )

        label_bar_renderer = HorizontalLabelBarRenderer(
            name="label_bar",
            data_streamer=label_streamer,
            grid_row=(2, 2),
            grid_column=(1, 1),
        )

        progress_renderer = ProgressRenderer(
            name="progress",
            data_streamer=progress_streamer,
            grid_row=(2, 2),
            grid_column=(1, 1),
        )

        video_renderer = RGBRenderer(
            name="video",
            data_streamer=vid_streamer,
            grid_row=(3, 3),
            grid_column=(1, 1),
        )

        # Set up orchestrator
        grid_rows = [40, 30, vid_streamer.size[1]]
        grid_cols = [vid_streamer.size[0]]
        orch = AnnotationOrchestrator(grid_rows, grid_cols, gap=0)

        orch.set_annotators(
            [vid_streamer, label_streamer, progress_streamer, title_streamer],
            [video_renderer, label_bar_renderer, progress_renderer, title_renderer],
            [
                ("video", "video"),
                ("labels", "label_bar"),
                ("progress", "progress"),
                ("title", "title"),
            ],
        )

        # Render
        os.makedirs("demo/output", exist_ok=True)
        outpath = "demo/output/context_style.mp4"
        orch.write(outpath, fps=30.0)
        print(f"Rendered: {outpath}")

    print("Returned to default styling")


def demo_predefined_styles():
    """Demonstrate predefined styles."""
    print("\n=== Predefined Styles Demo ===")

    styles = ["default", "dark", "minimal", "high_contrast"]

    for style_name in styles:
        print(f"\nApplying {style_name} style...")
        use_style(style_name)

        video_path = create_demo_video()
        label_path = create_demo_labels()

        # Set up streamers
        vid_streamer = VideoStreamer("video", video_path, sample_rate=30.0)
        label_streamer = StaticTabularStreamer(
            name="labels",
            data_source=label_path,
            data_col="label",
            time_col="time",
            sample_rate=30.0,
        )
        progress_streamer = ProgressStreamer("progress", label_streamer, sample_rate=30.0)
        title_streamer = StaticDataStreamer("title", f"{style_name.title()} Style")

        # Set up renderers
        title_renderer = StringRenderer(
            name="title",
            data_streamer=title_streamer,
            grid_row=(1, 1),
            grid_column=(1, 1),
        )

        label_bar_renderer = HorizontalLabelBarRenderer(
            name="label_bar",
            data_streamer=label_streamer,
            grid_row=(2, 2),
            grid_column=(1, 1),
        )

        progress_renderer = ProgressRenderer(
            name="progress",
            data_streamer=progress_streamer,
            grid_row=(2, 2),
            grid_column=(1, 1),
        )

        video_renderer = RGBRenderer(
            name="video",
            data_streamer=vid_streamer,
            grid_row=(3, 3),
            grid_column=(1, 1),
        )

        # Set up orchestrator
        grid_rows = [40, 30, vid_streamer.size[1]]
        grid_cols = [vid_streamer.size[0]]
        orch = AnnotationOrchestrator(grid_rows, grid_cols, gap=0)

        orch.set_annotators(
            [vid_streamer, label_streamer, progress_streamer, title_streamer],
            [video_renderer, label_bar_renderer, progress_renderer, title_renderer],
            [
                ("video", "video"),
                ("labels", "label_bar"),
                ("progress", "progress"),
                ("title", "title"),
            ],
        )

        # Render
        os.makedirs("demo/output", exist_ok=True)
        outpath = f"demo/output/{style_name}_style.mp4"
        orch.write(outpath, fps=30.0)
        print(f"Rendered: {outpath}")


if __name__ == "__main__":
    print("Vidplot Styling System Demo")
    print("=" * 50)

    # Run all demos
    demo_default_style()
    demo_custom_style()
    demo_context_manager()
    demo_predefined_styles()

    print("\n" + "=" * 50)
    print("All demos completed! Check the demo/output/ directory for results.")
    print("\nUsage examples:")
    print("  from vidplot import rc, use_style, set_font_size")
    print("  rc('font_scale', 0.8)  # Set individual parameter")
    print("  use_style('dark')      # Apply predefined style")
    print("  set_font_size(16)      # Use convenience function")
