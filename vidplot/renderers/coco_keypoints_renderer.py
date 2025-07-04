from typing import Any, Tuple
import cv2
import numpy as np

from vidplot.core import Renderer
from vidplot.style import rcParams


class COCOKeypointsRenderer(Renderer):
    """
    Renders COCO-format keypoints on a canvas within a bounding box.

    This renderer uses the global styling configuration as defaults and allows
    individual parameters to be overridden via **kwargs.

    Styling Parameters (can be overridden with **kwargs):
        color: RGB color for keypoint circles (tuple)
        radius: Radius of keypoint circles (int)
        thickness: Circle thickness, -1 for filled (int)
        draw_labels: Whether to draw labels on keypoints (bool)
        keypoint_labels: Mapping from index to string label (dict)
        font_scale: Font size scaling for labels (float)
        font_color: RGB color for label text (tuple)
        font_thickness: Font thickness for labels (int)
        font_face: OpenCV font face constant (int)
        confidence_threshold: Minimum confidence to show keypoints (float)
        assume_normalized: Force normalized/pixel input detection (bool or None)

    Global Style Integration:
        All parameters default to values from vidplot.style.rcParams().
        Use vidplot.style.rc() to change global defaults.
        Use vidplot.style.use_style() to apply predefined themes.

    Examples:
        # Use all global defaults
        renderer = COCOKeypointsRenderer("keypoints", streamer)

        # Override specific parameters
        renderer = COCOKeypointsRenderer("keypoints", streamer,
                                        color=(255, 0, 0),      # Red keypoints
                                        radius=5,               # Larger circles
                                        draw_labels=True,       # Show labels
                                        confidence_threshold=0.5)

        # Custom keypoint labels
        labels = {0: 'nose', 1: 'left_eye', 2: 'right_eye'}
        renderer = COCOKeypointsRenderer("keypoints", streamer,
                                        keypoint_labels=labels,
                                        font_scale=0.6)
    """

    def __init__(self, name: str, data_streamer, **kwargs):
        """
        Initialize COCOKeypointsRenderer with optional styling overrides.

        Args:
            name: Unique identifier for this renderer
            data_streamer: DataStreamer providing pose keypoints
            **kwargs: Optional styling parameter overrides:
                - color: RGB color for keypoints (default: from global config)
                - radius: Circle radius (default: from global config)
                - thickness: Circle thickness (default: from global config)
                - draw_labels: Whether to draw labels (default: False)
                - keypoint_labels: Index to label mapping (default: {})
                - font_scale: Font size scaling (default: from global config)
                - font_color: Label text color (default: from global config)
                - font_thickness: Label text thickness (default: from global config)
                - font_face: OpenCV font face (default: from global config)
                - confidence_threshold: Min confidence (default: from global config)
                - assume_normalized: Force input detection (default: None)

        Note:
            All kwargs override the corresponding global style parameters.
            See vidplot.style.rcParams() for current global defaults.
        """
        super().__init__(name, data_streamer)

        # Get default values from global style configuration
        config = rcParams()

        # Set styling parameters with kwargs override
        self.color = kwargs.get("color", config.marker_color)
        self.radius = kwargs.get("radius", config.marker_radius)
        self.thickness = kwargs.get("thickness", config.marker_thickness)
        self.draw_labels = kwargs.get("draw_labels", False)
        self.keypoint_labels = kwargs.get("keypoint_labels", {})
        self.font_scale = kwargs.get("font_scale", config.font_scale)
        self.font_color = kwargs.get("font_color", config.font_color)
        self.font_thickness = kwargs.get("font_thickness", config.font_thickness)
        self.font_face = kwargs.get("font_face", getattr(cv2, config.font_face))
        self.confidence_threshold = kwargs.get("confidence_threshold", config.confidence_threshold)
        self.assume_normalized = kwargs.get("assume_normalized", None)

    @property
    def _default_size(self):
        return (None, None)  # No fixed size, depends on the canvas

    def _is_normalized(self, pose: np.ndarray) -> bool:
        # Heuristic: if any keypoint is > 2, it's probably pixel-based
        if self.assume_normalized is not None:
            return self.assume_normalized
        return np.max(pose[:, :2]) <= 1.0

    def _render_pose(self, pose: np.ndarray, canvas: Any, bbox: Tuple[int, int, int, int]):
        x0, y0, w, h = bbox

        pose = np.asarray(pose)
        if pose.ndim != 2 or pose.shape[1] < 2:
            return  # Invalid pose shape

        is_norm = self._is_normalized(pose)
        keypoints = pose.copy()

        # Convert to pixel space
        if is_norm:
            keypoints[:, 0] = keypoints[:, 0] * w + x0
            keypoints[:, 1] = keypoints[:, 1] * h + y0
        else:
            keypoints[:, 0] += x0
            keypoints[:, 1] += y0

        for idx, kp in enumerate(keypoints):
            x, y = int(round(kp[0])), int(round(kp[1]))
            conf = kp[2] if kp.shape[0] > 2 else 1.0

            # Skip if confidence is too low or outside bbox
            if conf < self.confidence_threshold:
                continue
            if not (x0 <= x < x0 + w and y0 <= y < y0 + h):
                continue

            # Draw keypoint
            cv2.circle(canvas, (x, y), self.radius, self.color, self.thickness)

            # Optional label
            if self.draw_labels and idx in self.keypoint_labels:
                label = self.keypoint_labels[idx]
                cv2.putText(
                    canvas,
                    label,
                    (x + 2, y - 2),
                    self.font_face,
                    self.font_scale,
                    self.font_color,
                    self.font_thickness,
                    cv2.LINE_AA,
                )

    def _render(self, data: Any, bbox: Tuple[int, int, int, int], canvas: Any) -> Any:
        """
        Render COCO-style pose keypoints.

        Parameters:
        - data: One pose (np.ndarray of shape (K, 2 or 3)) or dict[int, pose]
        - bbox: Bounding box assumed to contain the full frame (x, y, w, h)
        - canvas: The canvas image to modify.

        Returns:
        - Modified canvas.
        """
        if data is None:
            return canvas

        if isinstance(data, dict):
            for pose in data.values():
                self._render_pose(pose, canvas, bbox)
        else:
            self._render_pose(data, canvas, bbox)

        return canvas
