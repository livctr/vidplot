"""
Visual test for painting a segmentation mask on an image.

This script loads an image, creates a dummy segmentation mask, and then uses
the `paint_segmentation_mask` utility to overlay the mask onto the image.
The "before" and "after" images are then displayed using matplotlib.
"""

import cv2
import numpy as np
import os

from vidplot import VideoCanvas
from vidplot.streamers import VideoStreamer, TimestampedDataStreamer
from vidplot.renderers import RGBRenderer, SegmentationRenderer
from vidplot.renderers.utils import get_tab10_color

from vidplot.renderers.segmentation_renderer import paint_segmentation_mask_in_place


def create_segmentation_data(image_path: str):
    """
    Loads an image and creates a corresponding dummy segmentation mask.

    Args:
        image_path: The path to the image file.

    Returns:
        A tuple containing the loaded image (as an RGB NumPy array) and the
        generated segmentation mask.

    Raises:
        FileNotFoundError: if the image cannot be loaded.
    """
    original_img_array = cv2.imread(image_path)
    if original_img_array is None:
        raise FileNotFoundError(
            f"Could not read image from {image_path}. Make sure the file exists."
        )

    # OpenCV reads images in BGR format, so convert to RGB for processing and display
    original_img_rgb = cv2.cvtColor(original_img_array, cv2.COLOR_BGR2RGB)

    img_height, img_width = original_img_rgb.shape[:2]

    # Create a dummy mask array with the same dimensions as the image
    dummy_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    # Define a rectangular area for the mask
    cv2.rectangle(dummy_mask, (50, 80), (250, 220), 1, -1)

    return original_img_rgb, dummy_mask


def test_image_segmentation():
    """Test that orchestrator handles duplicate names correctly."""
    try:
        image_path = os.path.join("./tests/input/giraffe.jpg")
        output_path = os.path.join("./tests/output/giraffe_seg.jpg")

        # Get the image and the segmentation mask
        original_image, segmentation_mask = create_segmentation_data(image_path)

        # Define a color for the mask in RGB format
        mask_color_rgb = (255, 0, 0)  # Red

        # Paint the segmentation mask on the image
        paint_segmentation_mask_in_place(
            original_image, segmentation_mask, alpha=0.6, color=mask_color_rgb
        )

        cv2.imwrite(output_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'giraffe.jpg' is located in the 'tests/input/' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    assert os.path.exists(output_path)


# --- Test Configuration ---
VIDEO_W, VIDEO_H = 640, 480
FPS = 30
DURATION_S = 5
NUM_FRAMES = DURATION_S * FPS


def create_dummy_video(output_path: str):
    """Creates a simple, blank video file for testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (VIDEO_W, VIDEO_H))

    for i in range(NUM_FRAMES):
        # Create a frame with a color that changes over time
        color_val = int(100 + 100 * (i / NUM_FRAMES))
        frame = np.full((VIDEO_H, VIDEO_W, 3), (color_val, color_val, color_val), dtype=np.uint8)
        writer.write(frame)

    writer.release()
    print(f"Dummy video created at: {output_path}")


def test_segmentation_rendering_pipeline():
    """
    Tests the end-to-end pipeline of rendering segmentations on a video.
    """
    # --- 1. Setup Paths and Create Test Data ---
    input_video_path = "./tests/input/videos/sample_video.mp4"
    segmentation_data_path = "./tests/input/frame_segmentations/segmentations.json"
    output_video_path = "./tests/output/video_with_segmentations.png"

    # Create dummy video for the test
    create_dummy_video(input_video_path)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Check if segmentation data exists
    if not os.path.exists(segmentation_data_path):
        raise FileNotFoundError(
            f"Segmentation data not found at {segmentation_data_path}. "
            "Please run `tests/scripts/generate_test_segmentations.py` first."
        )

    # --- 3. Setup Data Streamers ---
    video_streamer = VideoStreamer("video_stream", input_video_path)
    # Use TimestampedDataStreamer for segmentations
    segmentation_streamer = TimestampedDataStreamer(
        name="seg_stream",
        data_source=segmentation_data_path,
        time="frame_times",
        data="encoded_outputs",
    )

    # --- 4. Setup Renderers ---
    # Renderer for the base video
    video_renderer = RGBRenderer(
        name="base_video",
        data_streamer=video_streamer,
    )

    # Create a color map for the object IDs
    # The generation script uses IDs from 1 up to MAX_OBJECTS_PER_FRAME (5)
    id_to_color_map = {i: get_tab10_color(i) for i in range(1, 6)}

    # Renderer for the segmentation masks
    segmentation_renderer = SegmentationRenderer(
        name="seg_overlay",
        data_streamer=segmentation_streamer,
        id_to_color=id_to_color_map,
        alpha=0.3,
    )

    # --- 5. Orchestrate and Render ---
    height, width = video_streamer.size
    canvas = VideoCanvas(row_gap=0, col_gap=0)
    # Attach both renderers to the same cell (1,1)
    canvas.attach(
        video_streamer,
        video_renderer,
        grid_row=1,
        grid_col=1,
        height=[height],
        width=[width],
        z_index=0,
    )
    canvas.attach(
        segmentation_streamer,
        segmentation_renderer,
        grid_row=1,
        grid_col=1,
        height=[height],
        width=[width],
        z_index=1,
    )

    print("Starting rendering process...")
    canvas.write(output_video_path)
    print("Rendering complete.")

    # --- 6. Assert ---
    assert os.path.exists(output_video_path), f"Output video was not created at {output_video_path}"
    print(f"Successfully created output video at: {output_video_path}")


if __name__ == "__main__":
    # test_image_segmentation()
    test_segmentation_rendering_pipeline()
