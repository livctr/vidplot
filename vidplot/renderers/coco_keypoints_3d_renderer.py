from typing import Any, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from vidplot.core import Renderer


class COCOKeypoints3DRenderer(Renderer):
    """
    Renders 3D COCO-format keypoints inside a bounding box using matplotlib.

    This renderer uses the global styling configuration as defaults and allows
    individual parameters to be overridden via **kwargs.

    Styling Parameters (can be overridden with **kwargs):
        marker_color: Color for 3D keypoint markers (str)
        marker_size: Size of 3D keypoint markers (int)

    Global Style Integration:
        All parameters default to values from vidplot.style.rcParams().
        Use vidplot.style.rc() to change global defaults.
        Use vidplot.style.use_style() to apply predefined themes.

    Examples:
        # Use all global defaults
        renderer = COCOKeypoints3DRenderer("3d_keypoints", streamer)

        # Override specific parameters
        renderer = COCOKeypoints3DRenderer("3d_keypoints", streamer,
                                           marker_color='blue',      # Blue markers
                                           marker_size=30)          # Larger markers

        # Custom 3D view
        renderer = COCOKeypoints3DRenderer("3d_keypoints", streamer,
                                           figsize=(6, 6),           # Larger plot
                                           elev=45,                  # 45° elevation
                                           azim=-45)                # -45° azimuth
    """

    def __init__(
        self,
        name: str,
        data_streamer,
        figsize: Tuple[int, int] = (4, 4),
        elev: int = 10,
        azim: int = -90,
        confidence_threshold: float = 0.0,
        **kwargs,
    ):
        """
        Initialize COCOKeypoints3DRenderer with optional styling overrides.

        Args:
            name: Unique identifier for this renderer
            data_streamer: DataStreamer providing 3D pose data
            figsize: Figure size tuple for matplotlib
            elev: Elevation angle for 3D view
            azim: Azimuth angle for 3D view
            confidence_threshold: Minimum confidence to show keypoints
            **kwargs: Optional styling parameter overrides:
                - marker_color: Marker color string (default: 'red')
                - marker_size: Marker size (default: 20)

        Note:
            All kwargs override the corresponding global style parameters.
            See vidplot.style.rcParams() for current global defaults.
        """
        super().__init__(name, data_streamer)

        # Get default values from global style configuration
        # config = rcParams()

        # Set styling parameters with kwargs override
        self.marker_color = kwargs.get("marker_color", "red")
        self.marker_size = kwargs.get("marker_size", 20)

        # Store explicit parameters
        self.figsize = figsize
        self.elev = elev
        self.azim = azim
        self.confidence_threshold = confidence_threshold

    @property
    def _default_size(self):
        return (None, None)  # No fixed size, depends on the canvas

    def _render_3d_pose(self, pose: np.ndarray) -> np.ndarray:
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=self.elev, azim=self.azim)

        pose = np.asarray(pose)
        if pose.ndim != 2 or pose.shape[1] < 3:
            plt.close(fig)
            return None

        # Filter by confidence if present
        if pose.shape[1] == 4:
            conf = pose[:, 3]
        else:
            conf = np.ones(pose.shape[0])

        mask = conf >= self.confidence_threshold
        x, y, z = pose[mask, 0], pose[mask, 1], pose[mask, 2]

        ax.scatter(x, y, z, c=self.marker_color, s=self.marker_size)
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")

        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(int(h), int(w), 4)[..., :3]
        plt.close(fig)
        return img

    def _render(self, data: Any, bbox: Tuple[int, int, int, int], canvas: Any) -> Any:
        """
        Render 3D COCO-style pose keypoints.

        Parameters:
        - data: 3D pose data (np.ndarray of shape (K, 3 or 4))
        - bbox: Bounding box (x, y, width, height)
        - canvas: The canvas image to modify.

        Returns:
        - Modified canvas.
        """
        if data is None:
            return canvas

        x, y, w, h = bbox
        img = self._render_3d_pose(data)
        if img is None:
            return canvas

        img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        canvas[y : y + h, x : x + w] = img_resized
        return canvas
