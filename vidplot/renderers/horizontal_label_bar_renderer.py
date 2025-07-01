from typing import Any, Dict, List, Tuple
import numpy as np

from vidplot.core import Renderer


class HorizontalLabelBarRenderer(Renderer):
    """Useful for timestep-dependent labels in visualizations."""

    def __init__(
        self,
        name: str,
        data_streamer,
        label_to_color: Dict[int, Tuple[int, int, int]],
        grid_row: Tuple[int, int],
        grid_column: Tuple[int, int],
        z_index: int = 0,
        height: int = 20,
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
        self._label_to_color = label_to_color
        self._label_bar = None

    @property
    def _default_size(self):
        return (None, self._height)

    def _create_label_bar(
        self, labels: List[str], bar_height: int, bar_width: int
    ) -> Dict[str, tuple]:
        """Assign consistent BGR colors to label strings."""

        colors = [self._label_to_color[label] for label in labels]

        self._label_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

        total_samples = len(labels)
        segment_width = float(bar_width) / total_samples

        for i in range(len(labels)):
            start = int(i * segment_width)
            end = int((i + 1) * segment_width)
            self._label_bar[:, start:end] = colors[i]

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

        x1, y1, x2, y2 = bbox
        bar_width = x2 - x1
        bar_height = y2 - y1

        if bar_width <= 0 or bar_height <= 0:
            return canvas  # Nothing to draw

        if self._label_bar is None:
            self._create_label_bar(data, bar_height, bar_width)

        canvas[y1:y2, x1:x2] = self._label_bar
        return canvas
