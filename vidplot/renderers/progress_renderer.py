from typing import Any, Tuple
import cv2

from ..core.renderer import Renderer


class ProgressRenderer(Renderer):
    """
    Renders a horizontal progress bar as a vertical rectangle sweeping
    from left to right over the canvas, based on a float progress value.
    """

    def __init__(
        self,
        name: str,
        data_streamer,
        grid_row: Tuple[int, int],
        grid_column: Tuple[int, int],
        z_index: int = 0,
        bar_color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ):
        """
        Initializes the progress renderer.

        Parameters:
        - name: Unique name for the renderer
        - data_streamer: DataStreamer providing progress values
        - grid_row: Tuple of (start_row, end_row) in grid
        - grid_column: Tuple of (start_col, end_col) in grid
        - z_index: Depth ordering; larger values drawn on top
        - bar_color: Color of the moving progress bar (BGR)
        - thickness: Width of the vertical progress bar rectangle
        """
        super().__init__(name, data_streamer, grid_row, grid_column, z_index)
        self.bar_color = bar_color
        self.thickness = thickness

    @property
    def _default_size(self):
        return (None, None)  # No fixed size, depends on the canvas

    def _render(self, data: Any, bbox: Tuple[int, int, int, int], canvas: Any) -> Any:
        """
        Draws a vertical progress bar rectangle inside the given bounding box.

        Parameters:
        - data: A float between 0.0 and 1.0 indicating progress
        - bbox: Bounding box (x, y, width, height) to draw within
        - canvas: Full image canvas to draw on

        Returns:
        - The modified canvas with progress drawn
        """
        if data is None:
            return canvas

        if not isinstance(data, float):
            raise TypeError("ProgressRenderer expects a float between 0.0 and 1.0.")
        if not (-0.05 <= data <= 1.05):  # Allow a little tolerance
            raise ValueError("Progress value must be between 0.0 and 1.0.")
        data = max(0.0, min(1.0, data))  # Clamp to [0.0, 1.0]

        x, y, w, h = bbox
        progress_x = int(w * data)

        # Compute vertical bar position within the bounding box
        half_thick = self.thickness // 2
        bar_x1 = x + max(0, progress_x - half_thick)
        bar_x2 = x + min(w - 1, progress_x + half_thick)
        bar_y1 = y
        bar_y2 = y + h - 1

        # Draw vertical progress bar within the bbox
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), self.bar_color, thickness=-1)

        return canvas 