from typing import Any, Tuple, Optional

import cv2
import numpy as np

from vidplot.core import Renderer, DataStreamer


class RGBRenderer(Renderer):
    """
    Display RGB/BGR/grayscale frames with flexible resize modes and optional transparency
    Args:
        name: Unique renderer name.
        data_streamer: Supplies frame data.
        channel: 'rgb' or 'bgr'.
        resize_mode: 'stretch', 'fit', or 'center'.
        background: (R,G,B) background fill, or None for transparent.
    """

    def __init__(
        self,
        name: str,
        data_streamer: DataStreamer,
        channel: str = "rgb",
        resize_mode: str = "fit",
        background: Optional[Tuple[int, int, int]] = (0, 0, 0),
    ):
        super().__init__(name, data_streamer)
        assert channel in ("rgb", "bgr"), "channel must be 'rgb' or 'bgr'"
        assert resize_mode in (
            "stretch",
            "fit",
            "center",
        ), "resize_mode must be 'stretch', 'fit', or 'center'"
        self.channel = channel
        self.resize_mode = resize_mode
        self.background = background

    @property
    def _default_size(self) -> Tuple[int, int]:
        return (100, 100)

    def _convert_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2 or frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return frame if self.channel == "rgb" else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _calculate_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        # bbox: (x1, y1, x2, y2)
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        fh, fw = self.data_streamer.size[:2]
        if self.resize_mode == "stretch":
            return (x1, y1, x2, y2)
        if self.resize_mode == "fit":
            scale = min(w / fw, h / fh)
            nw, nh = int(fw * scale), int(fh * scale)
            ox = x1 + (w - nw) // 2
            oy = y1 + (h - nh) // 2
            return (ox, oy, ox + nw, oy + nh)
        # center
        ox = x1 + (w - fw) // 2
        oy = y1 + (h - fh) // 2
        return (ox, oy, ox + fw, oy + fh)

    def _render(
        self,
        data: Any,
        bbox: Tuple[int, int, int, int],
        canvas: Any,
    ) -> Any:
        if data is None:
            return canvas

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        frame = self._convert_to_rgb(np.array(data))
        # Background fill if needed
        if self.background is not None:
            canvas[y1:y2, x1:x2] = self.background
        # Resize frame to target region and blit
        resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        canvas[y1:y2, x1:x2] = resized
        return canvas
