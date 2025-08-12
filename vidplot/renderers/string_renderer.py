import cv2
import numpy as np
from typing import Any, Tuple, Optional
from vidplot.core import Renderer, DataStreamer
from vidplot.style import rcParams


class StringRenderer(Renderer):
    """
    Renders a string within a bounding box on an image canvas using OpenCV.
    Compatible with the new Renderer API and integrated with the styling system.

    This renderer uses the global styling configuration as defaults and allows
    individual parameters to be overridden via **kwargs.

    Styling Parameters (can be overridden with **kwargs):
        font_face: OpenCV font face constant (e.g., cv2.FONT_HERSHEY_SIMPLEX)
        font_scale: Font size scaling factor (float)
        font_color: Text color as RGB tuple (int, int, int)
        thickness: Font thickness (int)
        line_type: OpenCV line type (e.g., cv2.LINE_AA)
        num_expected_lines: Estimated number of text lines for size calculation (int)
        float_precision: Decimal places for float values, None to disable (int or None)

    Global Style Integration:
        All parameters default to values from vidplot.style.rcParams().
        Use vidplot.style.rc() to change global defaults.
        Use vidplot.style.use_style() to apply predefined themes.

    Examples:
        # Use all global defaults
        renderer = StringRenderer("text", streamer)

        # Override specific parameters
        renderer = StringRenderer("text", streamer,
                                 font_scale=1.5,
                                 font_color=(255, 0, 0),
                                 thickness=3)

        # Format float values
        renderer = StringRenderer("number", streamer,
                                 float_precision=2)  # "42.12"

        # Use global styling with context manager
        with vidplot.style.rc_context({'font_scale': 0.8}):
            renderer = StringRenderer("temp", streamer)
    """

    def __init__(self, name: str, data_streamer: DataStreamer, **kwargs):
        """
        Initialize StringRenderer with optional styling overrides.

        Args:
            name: Unique identifier for this renderer
            data_streamer: DataStreamer providing text content
            **kwargs: Optional styling parameter overrides:
                - font_face: OpenCV font face (default: from global config)
                - font_scale: Font size scaling (default: from global config)
                - font_color: RGB color tuple (default: from global config)
                - thickness: Font thickness (default: from global config)
                - line_type: OpenCV line type (default: from global config)
                - num_expected_lines: Lines for size calc (default: from global config)
                - float_precision: Decimal places for floats (default: from global config)

        Note:
            All kwargs override the corresponding global style parameters.
            See vidplot.style.rcParams() for current global defaults.
        """
        super().__init__(name, data_streamer)

        # Get default values from global style configuration
        config = rcParams()

        # Set styling parameters with kwargs override
        self.font_face = kwargs.get("font_face", getattr(cv2, config.font_face))
        self.font_scale = kwargs.get("font_scale", config.font_scale)
        self.font_color = kwargs.get("font_color", config.font_color)
        self.thickness = kwargs.get("thickness", config.font_thickness)
        self.line_type = kwargs.get("line_type", getattr(cv2, config.string_line_type))
        self.num_expected_lines = kwargs.get("num_expected_lines", config.string_num_expected_lines)
        self.float_precision = kwargs.get("float_precision", config.string_float_precision)

    @property
    def _default_size(self) -> Tuple[Optional[int], Optional[int]]:
        (_, text_h), _ = cv2.getTextSize("test", self.font_face, self.font_scale, self.thickness)
        # Width is flexible (None), height is estimated
        return (None, text_h * self.num_expected_lines)

    def _render(self, data: Any, bbox: Tuple[int, int, int, int], canvas: np.ndarray) -> np.ndarray:
        # If data is a dictionary, extract values; else treat it as text
        if isinstance(data, dict):
            text = data.get("text", "")
            font_face = data.get("font_face", self.font_face)
            font_scale = data.get("font_scale", self.font_scale)
            font_color = data.get("font_color", self.font_color)
            thickness = data.get("thickness", self.thickness)
            line_type = data.get("line_type", self.line_type)
        else:
            text = data
            font_face = self.font_face
            font_scale = self.font_scale
            font_color = self.font_color
            thickness = self.thickness
            line_type = self.line_type

        if text is None:
            return canvas

        if isinstance(text, float) and self.float_precision is not None:
            text = f"{text:.{self.float_precision}f}"
        else:
            text = str(text)

        x, y, x2, y2 = bbox
        _, h = x2 - x, y2 - y
        # Estimate text size
        (text_w, text_h), _ = cv2.getTextSize(text, font_face, font_scale, thickness)

        # Compute text origin (top-left of text baseline), adjusting for vertical fit
        text_x = x
        text_y = y + min(h, text_h)  # draw from top of box, max height is box height

        # Ensure the text does not go outside the canvas (clip coordinates)
        text_x = max(0, min(canvas.shape[1] - text_w, text_x))
        h = y2 - y
        text_y = y + (h + text_h) // 2  # centers vertically (OpenCV baseline adjustment)
        text_y = max(text_h, min(canvas.shape[0], text_y))

        # Draw text on the image
        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            font_face,
            font_scale,
            font_color,
            thickness,
            line_type,
        )

        return canvas
