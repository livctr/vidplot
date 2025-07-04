from typing import Any, Dict, List, Tuple, Optional
import cv2
import numpy as np

from vidplot.core import Renderer
from vidplot.streamers.label_bar_streamer import LabelBarStreamer
from vidplot.renderers.string_renderer import StringRenderer
from vidplot.style import rcParams


class LabelBarRenderer(Renderer):
    """
    Visualize time-stamp dependent labels with an integrated text renderer.

    This renderer uses the global styling configuration as defaults and allows
    individual parameters to be overridden via **kwargs.

    Styling Parameters (can be overridden with **kwargs):
        height: Height of the label bar in pixels (int)
        progress_bar_color: RGB color for the progress bar (tuple)
        progress_thickness: Thickness of the progress bar (int)
        font_scale: Font size scaling for text (float)
        font_color: RGB color for text (tuple)
        thickness: Font thickness for text (int)

    Global Style Integration:
        All parameters default to values from vidplot.style.rcParams().
        Use vidplot.style.rc() to change global defaults.
        Use vidplot.style.use_style() to apply predefined themes.

    Examples:
        # Use all global defaults with auto-generated colors
        renderer = LabelBarRenderer("labels", streamer)

        # Override specific parameters
        renderer = LabelBarRenderer("labels", streamer,
                                   height=30,                    # Taller bar
                                   progress_bar_color=(255, 0, 0), # Red progress
                                   progress_thickness=4)         # Thicker progress

        # Custom label colors
        label_to_color = {1: (255, 0, 0), 2: (0, 255, 0)}
        renderer = LabelBarRenderer("labels", streamer,
                                   label_to_color=label_to_color,
                                   font_scale=0.8)              # Larger text
    """

    def __init__(
        self,
        name: str,
        data_streamer: LabelBarStreamer,
        label_to_color: Optional[Dict[int, Tuple[int, int, int]]] = None,
        write_sampled_data_str: bool = True,
        **kwargs,
    ):
        """
        Initialize LabelBarRenderer with optional styling overrides.

        Args:
            name: Unique identifier for this renderer
            data_streamer: LabelBarStreamer providing label data
            label_to_color: Optional mapping from label IDs to RGB color tuples.
                           If None, colors will be auto-generated using the global color scheme.
            write_sampled_data_str: Whether to display the sampled label text on the bar
            **kwargs: Optional styling parameter overrides:
                - height: Bar height in pixels (default: from global config)
                - progress_bar_color: RGB progress bar color (default: from global config)
                - progress_thickness: Progress bar thickness (default: from global config)
                - font_scale: Font size scaling for text (default: from global config)
                - font_color: RGB text color (default: from global config)
                - thickness: Font thickness for text (default: from global config)

        Note:
            All kwargs override the corresponding global style parameters.
            The StringRenderer is automatically attached to handle text rendering.
            See vidplot.style.rcParams() for current global defaults.
        """
        super().__init__(name, data_streamer)

        # Get default values from global style configuration
        config = rcParams()

        # Set styling parameters with kwargs override
        self._height = kwargs.get("height", config.label_bar_height)
        self._progress_bar_color = kwargs.get("progress_bar_color", config.progress_bar_color)
        self._progress_thickness = kwargs.get("progress_thickness", config.progress_bar_thickness)

        # Store label color mapping (optional)
        self._label_to_color = label_to_color or {}

        # Initialize label bar cache
        self._label_bar = None

        # Flag for text rendering
        self._write_sampled_data_str = write_sampled_data_str

        # Create and attach text renderer for displaying sampled labels
        if self._write_sampled_data_str:
            self._text_renderer = StringRenderer(
                f"{name}_text",
                data_streamer,
                font_scale=kwargs.get("font_scale", config.font_scale),
                font_color=kwargs.get("font_color", config.font_color),
                thickness=kwargs.get("thickness", config.font_thickness),
            )

    @property
    def _default_size(self):
        return (None, self._height)

    def _get_color_for_label(self, label: int) -> Tuple[int, int, int]:
        """
        Get color for a label, using provided mapping or auto-generating from global scheme.

        Args:
            label: Label ID to get color for

        Returns:
            RGB color tuple for the label
        """
        if label in self._label_to_color:
            return self._label_to_color[label]

        # Auto-generate color from global color scheme
        config = rcParams()
        colors = config.label_bar_colors
        color_index = label % len(colors)
        return colors[color_index]

    def _create_label_bar(
        self, labels: List[str], bar_height: int, bar_width: int
    ) -> Dict[str, tuple]:
        """
        Create a label bar with consistent colors for each label segment.

        Args:
            labels: List of label IDs
            bar_height: Height of the bar in pixels
            bar_width: Width of the bar in pixels

        Returns:
            Dictionary mapping labels to their assigned colors
        """
        # Get colors for each label (auto-generate if not provided)
        colors = [self._get_color_for_label(label) for label in labels]

        # Create the label bar image
        self._label_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

        total_samples = len(labels)
        segment_width = float(bar_width) / total_samples

        # Fill each segment with its corresponding color
        for i in range(len(labels)):
            start = int(i * segment_width)
            end = int((i + 1) * segment_width)
            self._label_bar[:, start:end] = colors[i]

    def _render(
        self, data: Tuple[List[Any], Any, float], bbox: Tuple[int, int, int, int], canvas: Any
    ) -> Any:
        """
        Draw a horizontal label bar within the bounding box on the canvas.
        Each label's proportion is shown using a unique color, with an optional
        progress bar and text overlay.

        Args:
            data: Tuple of (uniform_data, sampled_data, progress)
                - uniform_data: List of label IDs for the full sequence
                - sampled_data: Current label ID
                - progress: Progress value between 0.0 and 1.0
            bbox: Bounding box (x, y, width, height) within which to draw
            canvas: The image canvas (numpy array) to draw on

        Returns:
            The modified canvas
        """
        if data is None:
            return canvas

        if len(data) != 3:
            raise ValueError("Ensure the streamer for LabelBarRenderer is LabelBarStreamer.")

        uniform_data, sampled_data, progress = data

        x1, y1, x2, y2 = bbox
        bar_width = x2 - x1
        bar_height = y2 - y1

        if bar_width <= 0 or bar_height <= 0:
            return canvas  # Nothing to draw

        # Create the label bar if not already created
        if self._label_bar is None:
            self._create_label_bar(uniform_data, bar_height, bar_width)

        # Draw the label bar onto the canvas
        canvas[y1:y2, x1:x2] = self._label_bar

        # Write str(sampled_data) on top of the label bar
        if self._write_sampled_data_str:
            canvas = self._text_renderer._render(sampled_data, (x1, y1, x2, y2), canvas)

        # Draw the vertical progress bar
        if not isinstance(progress, float):
            raise TypeError("LabelBarRenderer expects a float between 0.0 and 1.0.")
        if not (-0.05 <= progress <= 1.05):  # Allow a little tolerance
            raise ValueError("LabelBarRenderer: Progress value must be between 0.0 and 1.0.")

        # Clamp progress to valid range and calculate position
        progress = max(0.0, min(1.0, progress))
        progress_x = x1 + int(bar_width * progress)
        half_thick = self._progress_thickness // 2
        bar_x1 = max(x1, progress_x - half_thick)
        bar_x2 = min(x2, progress_x + half_thick)

        # Draw the progress bar
        cv2.rectangle(
            canvas,
            (bar_x1, y1),
            (bar_x2, y2),
            self._progress_bar_color,
            thickness=-1,  # Filled rectangle
        )

        return canvas
