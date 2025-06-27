from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from vidplot.core import Renderer


class HorizontalLabelBarRenderer(Renderer):
    """Useful for timestep-dependent labels in visualizations."""

    def __init__(
        self,
        name: str,
        data_streamer,
        grid_row: Tuple[int, int],
        grid_column: Tuple[int, int],
        z_index: int = 0,
        height: int = 20,
        color_seed: Optional[int] = None,
    ):
        """
        Parameters:
        - name: Unique name for the renderer
        - data_streamer: DataStreamer providing label data
        - grid_row: Tuple of (start_row, end_row) in grid
        - grid_column: Tuple of (start_col, end_col) in grid
        - z_index: Depth ordering; larger values drawn on top
        - height: Height of the label bar in pixels
        - color_seed: Optional seed to keep label colors consistent
        """
        super().__init__(name, data_streamer, grid_row, grid_column, z_index)
        self._height = height
        self._label_bar = None
        self._color_seed = color_seed if color_seed is not None else 42
        self._colors = None

    @property
    def _default_size(self):
        return (None, self._height)

    def _create_label_bar(
        self, labels: List[str], bar_height: int, bar_width: int
    ) -> Dict[str, tuple]:
        """Assign consistent BGR colors to label strings."""
        unique_labels = sorted(set(labels))
        n = len(unique_labels)
        if n > 10:
            raise ValueError("Too many unique labels for a bar plot (must be ≤ 10).")

        cmap = plt.get_cmap("tab10")
        colors = [tuple(int(255 * c) for c in to_rgb(cmap(i))) for i in range(n)]
        colors = [tuple(reversed(color)) for color in colors]  # Convert RGB to BGR
        self._colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        self._label_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

        total_samples = len(labels)
        segment_width = float(bar_width) / total_samples

        for i, label in enumerate(labels):
            color = self._colors[label]
            start = int(i * segment_width)
            end = int((i + 1) * segment_width)
            self._label_bar[:, start:end] = color

    def _render(self, data: List[str], bbox: Tuple[int, int, int, int], canvas: Any) -> Any:
        """
        Draw a horizontal label bar within the bounding box on the canvas.
        Each label's proportion is shown using a unique color.

        Parameters:
        - data: List of label strings
        - bbox: Bounding box (x, y, width, height) within which to draw the label bar
        - canvas: The image canvas (numpy array) to draw on

        Returns:
        - The modified canvas
        """
        if data is None:
            return canvas

        x_offset, y_offset, bar_width, bar_height = bbox

        if bar_width <= 0 or bar_height <= 0:
            return canvas  # Nothing to draw

        if self._label_bar is None:
            self._create_label_bar(data, bar_height, bar_width)

        canvas[y_offset : y_offset + bar_height, x_offset : x_offset + bar_width] = self._label_bar
        return canvas
