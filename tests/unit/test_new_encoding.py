#!/usr/bin/env python3
"""
Test script for the new segmentation encoding scheme.
"""

import numpy as np
import json
from pathlib import Path

from vidplot.encode.segmentations import (
    encode_segmentation_masks,
    decode_segmentation_masks,
    calculate_compression_ratio,
)


def create_test_masks():
    """Create test segmentation masks for demonstration."""
    # Create a few different test masks
    masks = []

    # Mask 1: Rectangle in the middle
    mask1 = np.zeros((100, 150), dtype=np.uint8)
    mask1[20:80, 30:120] = 1
    masks.append(mask1)

    # Mask 2: Circle
    mask2 = np.zeros((100, 150), dtype=np.uint8)
    y, x = np.ogrid[:100, :150]
    center_y, center_x = 50, 75
    radius = 25
    mask2[(x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2] = 1
    masks.append(mask2)

    # Mask 3: L-shaped object
    mask3 = np.zeros((100, 150), dtype=np.uint8)
    mask3[10:60, 10:40] = 1  # Vertical part
    mask3[10:30, 10:80] = 1  # Horizontal part
    masks.append(mask3)

    return masks


def test_full_encoding():
    """Test full encoding of multiple masks."""
    print("=== Testing Full Encoding ===")

    # Create test masks
    masks = create_test_masks()
    seg_ids = [1, 2, 3]  # IDs for each mask

    print(f"Created {len(masks)} test masks with IDs: {seg_ids}")
    for i, mask in enumerate(masks):
        print(f"  Mask {seg_ids[i]}: shape={mask.shape}, coverage={np.mean(mask):.2f}")

    # Encode all masks with full metadata (first frame)
    encoded_data = encode_segmentation_masks(seg_ids, masks, save_metadata=True)

    print(f"\nEncoded data keys: {list(encoded_data.keys())}")
    print(f"Shape: {encoded_data['shape']}")
    print(f"Dtype: {encoded_data['dtype']}")
    print(f"Total pixels: {encoded_data['total_pixels']}")
    print(f"Number of RLEs: {len(encoded_data['rles'])}")

    # Calculate compression ratio
    compression_ratio = calculate_compression_ratio(masks, encoded_data)
    print(f"Compression ratio: {compression_ratio:.2f}x")

    # Decode and verify
    decoded_ids, decoded_masks = decode_segmentation_masks(encoded_data)

    print(f"\nDecoded {len(decoded_masks)} masks with IDs: {decoded_ids}")

    # Verify reconstruction
    for i, (original, decoded) in enumerate(zip(masks, decoded_masks)):
        assert np.array_equal(original, decoded)

    return encoded_data


def test_subsequent_frame_encoding():
    """Test subsequent frame encoding for video sequences."""
    print("\n=== Testing Subsequent Frame Encoding ===")

    # Create reference data (from first frame)
    reference_masks = create_test_masks()
    reference_ids = [1, 2, 3]
    reference_data = encode_segmentation_masks(reference_ids, reference_masks, save_metadata=True)

    print("Reference data created from first frame")

    # Create "subsequent frame" masks (slightly modified)
    subsequent_masks = []
    for mask in reference_masks:
        # Create a slightly different mask (shifted by a few pixels)
        shifted_mask = np.roll(mask, (2, 3), axis=(0, 1))
        subsequent_masks.append(shifted_mask)

    subsequent_ids = [1, 2, 3]

    # Encode as subsequent frame without metadata
    subsequent_frame_data = encode_segmentation_masks(
        subsequent_ids, subsequent_masks, is_subsequent_frame=True, save_metadata=False
    )

    print(f"Subsequent frame encoded data keys: {list(subsequent_frame_data.keys())}")
    print(f"Number of RLEs: {len(subsequent_frame_data['rles'])}")

    # Calculate compression ratio
    compression_ratio = calculate_compression_ratio(subsequent_masks, subsequent_frame_data)
    print(f"Subsequent frame compression ratio: {compression_ratio:.2f}x")

    # Decode using reference data
    decoded_ids, decoded_masks = decode_segmentation_masks(subsequent_frame_data, reference_data)

    print(f"Decoded {len(decoded_masks)} masks with IDs: {decoded_ids}")

    # Verify reconstruction
    for i, (original, decoded) in enumerate(zip(subsequent_masks, decoded_masks)):
        assert np.array_equal(original, decoded)


def test_video_sequence():
    """Test encoding a video sequence with multiple frames."""
    print("\n=== Testing Video Sequence ===")

    # Simulate a video sequence with 5 frames
    num_frames = 5
    frame_size = (100, 150)

    # Create a moving circle across frames
    video_masks = []
    video_ids = []

    for frame_idx in range(num_frames):
        # Create a circle that moves across the frame
        mask = np.zeros(frame_size, dtype=np.uint8)
        y, x = np.ogrid[: frame_size[0], : frame_size[1]]

        # Circle moves from left to right
        center_x = 30 + frame_idx * 20
        center_y = 50
        radius = 20

        mask[(x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2] = 1
        video_masks.append(mask)
        video_ids.append(frame_idx + 1)

    print(f"Created video sequence with {num_frames} frames")

    # Encode first frame with full metadata
    first_frame_data = encode_segmentation_masks(
        [video_ids[0]], [video_masks[0]], save_metadata=True
    )

    print("First frame encoded with full metadata")
    print(f"  Size: {len(str(first_frame_data))} characters")

    # Encode subsequent frames without metadata
    subsequent_frames_data = []
    for i in range(1, num_frames):
        frame_data = encode_segmentation_masks(
            [video_ids[i]], [video_masks[i]], is_subsequent_frame=True, save_metadata=False
        )
        subsequent_frames_data.append(frame_data)
        print(f"Frame {i+1} encoded without metadata")
        print(f"  Size: {len(str(frame_data))} characters")

    # Calculate total compression
    total_original_size = sum(mask.nbytes for mask in video_masks)
    total_encoded_size = len(str(first_frame_data)) + sum(
        len(str(data)) for data in subsequent_frames_data
    )
    total_compression = total_original_size / total_encoded_size

    print(f"\nTotal compression ratio: {total_compression:.2f}x")
    print(f"Original size: {total_original_size} bytes")
    print(f"Encoded size: ~{total_encoded_size} characters")


def save_encoded_data(encoded_data, filename):
    """Save encoded data to file."""
    output_path = Path(f"tests/output/{filename}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(encoded_data, f, indent=2)

    print(f"Saved encoded data to: {output_path}")


def main():
    """Run all encoding tests."""
    print("New Segmentation Encoding Scheme - Test Suite")
    print("=" * 60)

    try:
        # Test full encoding
        full_data = test_full_encoding()
        save_encoded_data(full_data, "full_encoding.json")

        # Test subsequent frame encoding
        subsequent_data, ref_data = test_subsequent_frame_encoding()
        save_encoded_data(subsequent_data, "subsequent_frame_encoding.json")
        save_encoded_data(ref_data, "reference_data.json")

        # Test video sequence
        test_video_sequence()

        print("\n" + "=" * 60)
        print("All encoding tests completed successfully!")
        print("Check the 'tests/output/' directory for encoded data files.")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
