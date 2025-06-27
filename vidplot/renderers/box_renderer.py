from typing import Any, Tuple
import cv2

from vidplot.core import Renderer


class BoxRenderer(Renderer):
    """
    Renders bounding boxes inside a given bounding box region on a canvas.
    The rendered boxes are defined by normalized or absolute (x, y, w, h) coordinates.
    """

    def __init__(
        self,
        name: str,
        data_streamer,
        grid_row: Tuple[int, int],
        grid_column: Tuple[int, int],
        z_index: int = 0,
        color: Tuple[int, int, int] = (0, 255, 0),  # Default green
        thickness: int = 2,
    ):
        """
        Parameters:
        - name: Unique name for the renderer
        - data_streamer: DataStreamer providing (x, y, w, h) for boxes
        - grid_row: Tuple of (start_row, end_row) in grid
        - grid_column: Tuple of (start_col, end_col) in grid
        - z_index: Depth ordering; larger values drawn on top
        - color: BGR color of the box
        - thickness: Border thickness
        """
        super().__init__(name, data_streamer, grid_row, grid_column, z_index)
        self.default_color = color
        self.default_thickness = thickness

    @property
    def _default_size(self):
        return (None, None)  # No fixed size, depends on the canvas

    def _render(self, data: Any, bbox: Tuple[int, int, int, int], canvas: Any) -> Any:
        """
        Renders bounding boxes relative to the given bounding box area.
        Assumes the provided bounding box is generated.

        Parameters:
        - data: A list of dicts with keys: "box", "score", "id", "text_label".
                "box" is (x1, y1, x2, y2), normalized or absolute.
        - bbox: The parent bounding box (x1, y1, x2, y2).
        - canvas: The image to draw on.

        Returns:
        - The modified canvas.
        """
        if data is None:
            return canvas

        if not isinstance(data, list):
            raise ValueError("Expected data to be a list of dictionaries.")

        x01, y01, x02, y02 = bbox
        w0 = x02 - x01
        h0 = y02 - y01

        for entry in data:
            box = entry.get("box")
            if box is None:
                continue

            if not isinstance(box, (tuple, list)) or len(box) != 4:
                raise ValueError("Each 'box' must be a tuple of (x1, y1, x2, y2).")

            x1, y1, x2, y2 = box
            is_normalized = all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2))

            if not is_normalized:
                x1 = float(x1) / w0
                y1 = float(y1) / h0
                x2 = float(x2) / w0
                y2 = float(y2) / h0

            x1 = int(x01 + x1 * w0)
            y1 = int(y01 + y1 * h0)
            x2 = int(x01 + x2 * w0)
            y2 = int(y01 + y2 * h0)

            # Draw bounding box
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                self.default_color,
                self.default_thickness,
            )

            # Prepare annotation text
            label_parts = []
            if "text_label" in entry:
                label_parts.append(f"{entry['text_label']} ")
            if "id" in entry:
                label_parts.append(f"id:{entry['id']}")
            if "score" in entry:
                label_parts.append(f"{entry['score']:.2f}")
            label = " ".join(label_parts)

            if label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_origin = (x1, y1 - 5 if y1 - 5 > 10 else y1 + 15)
                cv2.putText(
                    canvas,
                    label,
                    text_origin,
                    font,
                    font_scale,
                    self.default_color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

        return canvas
