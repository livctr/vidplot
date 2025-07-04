from typing import Any, Tuple, Dict, Optional
import cv2
import numpy as np

from vidplot.core import Renderer, DataStreamer
from vidplot.encode.segmentations import decode_segmentation_masks


def paint_segmentation_mask_in_place(image_array, mask_array, alpha=0.5, color=(255, 0, 0)):
    """
    Overlays a segmentation mask on an image IN PLACE with a specified transparency and color.
    - Non-segmented areas remain identical to the original image (which is image_array itself).
    - Segmented areas are blended as: (mask_color * alpha) + (image_pixel * (1 - alpha)).

    Args:
        image_array (numpy.ndarray): The original image as a (H, W, 3) NumPy array (assumed RGB).
                                     This array will be modified directly.
        mask_array (numpy.ndarray): The segmentation mask as a (H, W) NumPy array.
                                    Pixels with a value > 0 are considered part of the mask.
        alpha (float): Transparency of the mask within the segmented area.
                       0.0 = segmented area is original image color (fully transparent mask)
                       1.0 = segmented area is fully the mask color (fully opaque mask)
                       A value like 0.3 to 0.7 usually works well for an overlay.
        color (tuple): RGB color of the mask (e.g., (255, 0, 0) for red).

    Returns:
        None: The function modifies `image_array` directly.
    """
    # 1. Image preparation (no copy, direct use of image_array)
    # Assume image_array is already RGB.

    # 2. Process the mask
    # Ensure mask dimensions match image dimensions
    if mask_array.shape != image_array.shape[:2]:
        raise ValueError(
            f"Mask dimensions {mask_array.shape} do not match image dimensions "
            f"{image_array.shape[:2]}"
        )

    # Create boolean mask: True where mask_array > 0
    segmentation_area = mask_array > 0

    # 3. Prepare the overlay color
    # The 'color' argument is assumed to be RGB.
    overlay_color_rgb = np.array(color, dtype=np.uint8)

    # 4. Perform the blending ONLY on the segmented area
    # Convert relevant parts to float for accurate blending calculation (0-1 range)
    # Original image pixels in the segmented area (from image_array directly)
    image_pixels_in_mask_float = image_array[segmentation_area].astype(np.float32) / 255.0

    # Overlay color (repeated for each channel) - ensure it's also float 0-1
    overlay_color_float = overlay_color_rgb.astype(np.float32) / 255.0

    # Apply the blending formula: (mask_color * alpha) + (image_pixel * (1 - alpha))
    blended_pixels_float = overlay_color_float * alpha + image_pixels_in_mask_float * (1 - alpha)

    # Convert back to 0-255 uint8 and assign back to the image_array
    # Clip values to ensure they stay within [0, 1] before scaling to [0, 255]
    image_array[segmentation_area] = (np.clip(blended_pixels_float, 0, 1) * 255).astype(np.uint8)


class SegmentationRenderer(Renderer):
    """
    Overlay segmentation masks onto a rendered frame.

    Intended to be attached to an RGBRenderer, so it reuses its bbox.

    Args:
        name: Unique renderer name.
        data_streamer: Supplies segmentation mask (2D array of IDs).
        id_to_color: Optional mapping from integer IDs to BGR colors.
                     If None, random distinct colors are generated.
        alpha: Opacity of the overlay, between 0.0 (transparent) and 1.0 (opaque).
    """

    def __init__(
        self,
        name: str,
        data_streamer: DataStreamer,
        id_to_color: Optional[Dict[int, Tuple[int, int, int]]] = None,
        alpha: float = 0.5,
    ):
        super().__init__(name, data_streamer)
        self.id_to_color = id_to_color or {}
        self.alpha = alpha

    @property
    def _default_size(self) -> Tuple[int, int]:
        # fallback if no parent bbox is given
        return (100, 100)

    def _render(
        self,
        data: Any,
        bbox: Tuple[int, int, int, int],
        canvas: Any,
    ) -> Any:
        mask = np.array(data)
        x1, y1, x2, y2 = bbox
        target_w, target_h = x2 - x1, y2 - y1

        if data is None:
            return canvas

        # If this is the first frame with metadata, store it for future frames
        if "shape" in data:
            self.reference_data = data
        elif self.reference_data is None:
            # We need reference data to decode subsequent frames
            return canvas

        # Decode the masks for the current frame
        seg_ids, seg_masks = decode_segmentation_masks(data, self.reference_data)

        # Get original frame dimensions from reference data
        fh, fw = self.reference_data["shape"]

        # Process each mask
        for seg_id, mask in zip(seg_ids, seg_masks):
            if seg_id not in self.id_to_color:
                continue

            color = self.id_to_color[seg_id]

            resized_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            target_view = canvas[y1:y2, x1:x2, :]
            paint_segmentation_mask_in_place(
                target_view, resized_mask, alpha=self.alpha, color=color
            )

        return canvas
