from typing import Any, Tuple, Dict
import cv2
import numpy as np

from vidplot.core import Renderer, DataStreamer
from vidplot.style import rcParams


def paint_box_in_place(
    image_array: np.ndarray,
    bbox_coords: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    label: str = None,
    thickness: int = 2,
    font_scale: float = 0.5,
):
    """
    Draws a bounding box and an optional label on an image in place.

    Args:
        image_array: The NumPy array of the image to draw on.
        bbox_coords: A tuple (x1, y1, x2, y2) for the box coordinates.
        color: The RGB color for the box and label background.
        label: The text to display on the label.
        thickness: The thickness of the box lines.
        font_scale: The scale of the label font.
    """
    x1, y1, x2, y2 = bbox_coords

    # Draw the main bounding box rectangle
    cv2.rectangle(image_array, (x1, y1), (x2, y2), color, thickness)

    if label:
        # Set up font and calculate the size of the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Create a filled rectangle for the label's background
        label_bg_y2 = y1
        label_bg_y1 = y1 - text_height - baseline

        # Ensure the label does not go off the top of the screen
        label_bg_y1 = max(label_bg_y1, 0)

        cv2.rectangle(image_array, (x1, label_bg_y1), (x1 + text_width, label_bg_y2), color, -1)

        # Put the label text on top of the background
        # Use white text for better contrast against a colored background
        text_y = y1 - baseline // 2
        cv2.putText(image_array, label, (x1, text_y), font, font_scale, (255, 255, 255), 1)


class BoxRenderer(Renderer):
    """
    Overlay bounding boxes on a video frame, aligned with a base renderer's bbox.

    This renderer uses the global styling configuration as defaults and allows
    individual parameters to be overridden via **kwargs.

    Styling Parameters (can be overridden with **kwargs):
        line_thickness: Thickness of box borders (int)
        font_scale: Font size scaling for labels (float)

    Global Style Integration:
        All parameters default to values from vidplot.style.rcParams().
        Use vidplot.style.rc() to change global defaults.
        Use vidplot.style.use_style() to apply predefined themes.

    Examples:
        # Use all global defaults
        renderer = BoxRenderer("boxes", streamer, id_to_color)

        # Override specific parameters
        renderer = BoxRenderer("boxes", streamer, id_to_color,
                              line_thickness=3,
                              font_scale=0.8)

        # Use different coordinate formats
        renderer = BoxRenderer("boxes", streamer, id_to_color,
                              box_representation_format="xywh",
                              resize_mode="fit")
    """

    def __init__(
        self,
        name: str,
        data_streamer: DataStreamer,
        id_to_color: Dict[int, Tuple[int, int, int]],
        label_box: bool = True,
        box_representation_format: str = "xyxy",
        resize_mode: str = "fit",
        **kwargs,
    ):
        """
        Initialize BoxRenderer with optional styling overrides.

        Args:
            name: Unique identifier for this renderer
            data_streamer: DataStreamer providing box data
            id_to_color: Mapping from box IDs to RGB color tuples
            label_box: Whether to draw labels on boxes
            box_representation_format: 'xyxy' or 'xywh' format
            resize_mode: 'stretch', 'fit', or 'center' for coordinate mapping
            **kwargs: Optional styling parameter overrides:
                - line_thickness: Box border thickness (default: from global config)
                - font_scale: Font size scaling (default: from global config)

        Note:
            All kwargs override the corresponding global style parameters.
            See vidplot.style.rcParams() for current global defaults.
        """
        super().__init__(name, data_streamer)

        # Get default values from global style configuration
        config = rcParams()

        # Set styling parameters with kwargs override
        self.line_thickness = kwargs.get("line_thickness", config.box_thickness)
        self.font_scale = kwargs.get("font_scale", config.font_scale)

        # Store explicit parameters
        self.label_box = label_box
        self.box_representation_format = box_representation_format
        self.resize_mode = resize_mode

        # Validate parameters
        assert self.resize_mode in ("stretch", "fit", "center"), "Invalid resize_mode"
        assert self.box_representation_format in ("xyxy", "xywh"), "Invalid format"

        # Store color mapping
        self.id_to_color = id_to_color

    @property
    def _default_size(self) -> Tuple[int, int]:
        return (100, 100)

    def _calculate_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        return bbox

    def _render(
        self,
        data: Any,
        bbox: Tuple[int, int, int, int],
        canvas: Any,
    ) -> Any:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if data is None or not data.get("boxes"):
            return canvas

        if not hasattr(self, "_cached_shape"):
            self._cached_shape = data["shape"]
        fh, fw = self._cached_shape

        for item in data["boxes"]:
            coords = item["box"]
            box_id = item.get("id")
            if box_id not in self.id_to_color:
                continue
            color = self.id_to_color[box_id]

            if self.box_representation_format == "xywh":
                x, y, bw, bh = coords
                x2b, y2b = x + bw, y + bh
                x1b, y1b = x, y
            else:
                x1b, y1b, x2b, y2b = coords

            # map box coords based on resize_mode
            if self.resize_mode == "stretch":
                xs, ys = w / fw, h / fh
                rx1, ry1 = int(x1b * xs), int(y1b * ys)
                rx2, ry2 = int(x2b * xs), int(y2b * ys)
            elif self.resize_mode == "fit":
                scale = min(w / fw, h / fh)
                nw, nh = int(fw * scale), int(fh * scale)
                x_off = (w - nw) // 2
                y_off = (h - nh) // 2
                rx1 = int(x1b * scale + x_off)
                ry1 = int(y1b * scale + y_off)
                rx2 = int(x2b * scale + x_off)
                ry2 = int(y2b * scale + y_off)
            else:  # center
                x_off = (w - fw) // 2
                y_off = (h - fh) // 2
                rx1 = x1b + x_off
                ry1 = y1b + y_off
                rx2 = x2b + x_off
                ry2 = y2b + y_off
                rx1, ry1 = max(0, rx1), max(0, ry1)
                rx2, ry2 = min(w, rx2), min(h, ry2)
                if rx1 >= rx2 or ry1 >= ry2:
                    continue

            abs_box = (rx1 + x1, ry1 + y1, rx2 + x1, ry2 + y1)
            label = None
            if self.label_box:
                score = item.get("score")
                label = f"ID: {box_id}"
                if score is not None:
                    label += f" ({score:.2f})"

            paint_box_in_place(
                canvas,
                abs_box,
                color=color,
                label=label,
                thickness=self.line_thickness,
                font_scale=self.font_scale,
            )

        return canvas
