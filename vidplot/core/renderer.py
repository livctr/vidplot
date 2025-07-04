from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List

from .streamer import DataStreamer, SizedStreamerProtocol


class Renderer(ABC):
    """
    Abstract base for all renderers.

    A Renderer draws streamed data onto a canvas. It supports:
      - defining a default or streamed size,
      - transforming bounding boxes via `_calculate_bbox`,
      - performing the main draw logic in `_render`,
      - attaching child renderers that run sequentially after the parent.

    Example:
        parent = VideoRenderer("video", video_streamer)
        overlay = OverlayRenderer("overlay", overlay_streamer)
        parent.attach(overlay)

        # In the render loop:
        canvas = parent.render(frame, bbox, canvas)
    """

    def __init__(
        self,
        name: str,
        data_streamer: DataStreamer,
    ) -> None:
        """
        Initialize a Renderer.

        Args:
            name: Identifier for this renderer.
            data_streamer: Supplies the data payloads.

        Attributes:
            _attached_renderers: List of child renderers to invoke after this one.
        """
        self.name = name
        self.data_streamer = data_streamer
        self._attached_renderers: List[Renderer] = []

    @property
    @abstractmethod
    def _default_size(self) -> Optional[Tuple[Optional[int], Optional[int]]]:
        """
        The fallback (width, height) when the streamer has no defined size.

        Subclasses must override this to declare their natural dimensions.
        """
        raise NotImplementedError(
            "Renderers without a sized data streamer must implement _default_size()"
        )

    @property
    def default_size(self) -> Optional[Tuple[Optional[int], Optional[int]]]:
        """
        Determine the effective render size.

        Returns:
            - If `data_streamer` implements `SizedStreamerProtocol`, returns its `size`.
            - Otherwise returns the subclass-defined `_default_size`.
        """
        if isinstance(self.data_streamer, SizedStreamerProtocol):
            return self.data_streamer.size
        return self._default_size

    def attach(self, renderer: "Renderer") -> None:
        """
        Attach a child renderer.

        The attached renderer will be invoked after this renderer's draw,
        receiving the transformed bounding box.

        Args:
            renderer: Another Renderer instance to chain.
        """
        self._attached_renderers.append(renderer)

    def detach(self, renderer: "Renderer") -> None:
        """
        Remove a previously attached child renderer.

        Args:
            renderer: The renderer to remove from the attachment list.
        """
        if renderer in self._attached_renderers:
            self._attached_renderers.remove(renderer)

    def _calculate_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Transform the input bounding box.

        Default behavior is identity. Override to implement offsets,
        scaling, or alignment relative to another renderer.

        Args:
            bbox: (x, y, width, height) of the region to draw into.

        Returns:
            A modified (x, y, width, height) bbox.
        """
        return bbox

    @abstractmethod
    def _render(
        self,
        data: Any,
        bbox: Tuple[int, int, int, int],
        canvas: Any,
    ) -> Any:
        """
        Perform the actual drawing onto `canvas`.

        Args:
            data: The payload from `data_streamer` (e.g., image frame, text).
            bbox: Region (x, y, width, height) within `canvas` to draw.
            canvas: Drawing surface (e.g., a numpy image).

        Returns:
            The modified `canvas` with new drawing applied.
        """
        raise NotImplementedError("Subclasses must implement _render()")

    def render(
        self,
        data: Any,
        bbox: Tuple[int, int, int, int],
        canvas: Any,
    ) -> Any:
        """
        High-level render flow:

        1. If `data` is None, skip all drawing.
        2. Transform `bbox` via `_calculate_bbox`.
        3. Draw via `_render(data, transformed_bbox, canvas)`.
        4. For each child in `_attached_renderers`, call its
            `render(data, transformed_bbox, canvas)`.

        Args:
            data: The value produced by the associated `DataStreamer`.
            bbox: Base region to draw into (x, y, width, height).
            canvas: The image or surface being mutated.

        Returns:
            The final `canvas` after all render steps.
        """
        # Skip if there's nothing to draw
        if data is None:
            return canvas

        # Parent-level bbox transform
        transformed_bbox = self._calculate_bbox(bbox)

        # Draw this renderer's layer
        canvas = self._render(data, transformed_bbox, canvas)

        # Chain child renderers
        for child in self._attached_renderers:
            canvas = child.render(data, transformed_bbox, canvas)

        return canvas
