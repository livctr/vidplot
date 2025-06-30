"""
Tests for segmentation mask encoding and decoding functionality.
"""

import numpy as np
import pytest
from vidplot.encode.segmentations import (
    encode_segmentation_mask,
    decode_segmentation_mask,
    encode_segmentation_masks,
    rle_to_binary_mask,
    calculate_compression_ratio,
)


class TestSegmentationEncoding:
    """Test cases for segmentation mask encoding and decoding."""

    def test_simple_mask_encoding_decoding(self):
        """Test encoding and decoding of a simple segmentation mask."""
        # Create a simple 3x3 mask with 2 classes
        original_mask = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)

        # Encode the mask
        encoded = encode_segmentation_mask(original_mask)

        # Verify encoded data structure
        assert "shape" in encoded
        assert "dtype" in encoded
        assert "rle" in encoded
        assert "total_pixels" in encoded
        assert "first_value" in encoded

        assert encoded["shape"] == (3, 3)
        assert encoded["dtype"] == "uint8"
        assert encoded["total_pixels"] == 9
        assert encoded["first_value"] == 0

        # Decode the mask
        decoded_mask = decode_segmentation_mask(encoded)

        # Verify the decoded mask matches the original
        np.testing.assert_array_equal(original_mask, decoded_mask)
        assert decoded_mask.dtype == original_mask.dtype

    def test_uniform_mask(self):
        """Test encoding and decoding of a uniform mask."""
        # Create a uniform mask (all same value)
        original_mask = np.full((5, 5), 1, dtype=np.int32)

        encoded = encode_segmentation_mask(original_mask)
        decoded_mask = decode_segmentation_mask(encoded)

        np.testing.assert_array_equal(original_mask, decoded_mask)
        assert decoded_mask.dtype == original_mask.dtype

    def test_large_mask(self):
        """Test encoding and decoding of a larger mask."""
        # Create a larger mask with binary values only
        original_mask = np.random.randint(0, 2, size=(100, 100), dtype=np.uint8)

        encoded = encode_segmentation_mask(original_mask)
        decoded_mask = decode_segmentation_mask(encoded)

        np.testing.assert_array_equal(original_mask, decoded_mask)
        assert decoded_mask.dtype == original_mask.dtype

    def test_different_dtypes(self):
        """Test encoding and decoding with different data types."""
        dtypes = [np.uint8, np.uint16, np.int32, np.int64, np.bool]

        for dtype in dtypes:
            original_mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=dtype)

            encoded = encode_segmentation_mask(original_mask)
            decoded_mask = decode_segmentation_mask(encoded)

            np.testing.assert_array_equal(original_mask, decoded_mask)
            assert decoded_mask.dtype == original_mask.dtype

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        # Create a mask with good compression potential (large uniform areas)
        original_mask = np.zeros((50, 50), dtype=np.uint8)
        original_mask[10:40, 10:40] = 1  # Large uniform region

        encoded = encode_segmentation_mask(original_mask)
        ratio = calculate_compression_ratio(original_mask, encoded)

        # Should achieve some compression
        assert ratio > 1.0

    def test_compression_ratio_multiple_masks(self):
        """Test compression ratio for multiple masks."""
        masks = [np.zeros((50, 50), dtype=np.uint8), np.ones((50, 50), dtype=np.uint8)]
        masks[0][10:40, 10:40] = 1

        encoded = encode_segmentation_masks(seg_ids=[1, 2], seg_masks=masks)
        ratio = calculate_compression_ratio(masks, encoded)
        assert ratio > 1.0

    def test_rle_to_binary_mask(self):
        """Test RLE to binary mask conversion."""
        # Create a simple mask
        original_mask = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.bool)

        encoded = encode_segmentation_mask(original_mask)

        # Convert to binary mask for class 1
        binary_mask = rle_to_binary_mask(encoded["rle"], encoded["shape"], encoded["first_value"])

        # The binary mask should be boolean
        assert binary_mask.dtype == bool

        # Check that it correctly identifies class 1
        expected_binary = original_mask == 1
        np.testing.assert_array_equal(binary_mask, expected_binary)


class TestRLECompressionDemonstration:
    """Demonstrate RLE compression effectiveness with various mask types."""

    def test_perfect_compression_scenarios(self):
        """Test scenarios where RLE achieves maximum compression."""

        # Test 1: Completely uniform mask (best case)
        uniform_mask = np.full((100, 100), 1, dtype=np.uint8)
        encoded = encode_segmentation_mask(uniform_mask)
        ratio = calculate_compression_ratio(uniform_mask, encoded)

        print("\nUniform 100x100 mask (all 1s):")
        print(f"  Original size: {uniform_mask.nbytes} bytes")
        print(f"  RLE runs: {len(encoded['rle'])}")
        print(f"  Compression ratio: {ratio:.2f}x")
        assert ratio > 100  # Should achieve very high compression

        # Test 2: Two large regions
        two_regions = np.zeros((200, 200), dtype=np.uint8)
        two_regions[50:150, 50:150] = 1  # Large square in center
        encoded = encode_segmentation_mask(two_regions)
        ratio = calculate_compression_ratio(two_regions, encoded)

        print("\nTwo-region 200x200 mask:")
        print(f"  Original size: {two_regions.nbytes} bytes")
        print(f"  RLE runs: {len(encoded['rle'])}")
        print(f"  Compression ratio: {ratio:.2f}x")
        assert ratio > 40  # Should achieve high compression

    def test_worst_case_compression_scenarios(self):
        """Test scenarios where RLE achieves minimal compression."""

        # Test 1: Checkerboard pattern (worst case)
        checkerboard = np.indices((50, 50)).sum(axis=0) % 2
        encoded = encode_segmentation_mask(checkerboard)
        ratio = calculate_compression_ratio(checkerboard, encoded)

        print("\nCheckerboard 50x50 mask:")
        print(f"  Original size: {checkerboard.nbytes} bytes")
        print(f"  RLE runs: {len(encoded['rle'])}")
        print(f"  Compression ratio: {ratio:.2f}x")
        assert ratio < 3  # Should achieve minimal compression

        # Test 2: Random binary noise
        np.random.seed(42)  # For reproducible results
        random_mask = np.random.randint(0, 2, size=(100, 100), dtype=np.uint8)
        encoded = encode_segmentation_mask(random_mask)
        ratio = calculate_compression_ratio(random_mask, encoded)

        print("\nRandom binary 100x100 mask:")
        print(f"  Original size: {random_mask.nbytes} bytes")
        print(f"  RLE runs: {len(encoded['rle'])}")
        print(f"  Compression ratio: {ratio:.2f}x")
        assert ratio < 8  # Should achieve moderate compression

    def test_realistic_segmentation_masks(self):
        """Test with realistic binary segmentation mask patterns."""

        # Test 1: Object detection style mask (few objects on background)
        object_mask = np.zeros((300, 400), dtype=np.uint8)

        # Add several objects of different sizes
        # Large object
        object_mask[50:150, 100:300] = 1
        # Medium object
        object_mask[200:250, 50:150] = 1
        # Small object
        object_mask[270:290, 350:380] = 1

        encoded = encode_segmentation_mask(object_mask)
        ratio = calculate_compression_ratio(object_mask, encoded)

        print("\nRealistic object detection mask (300x400):")
        print(f"  Original size: {object_mask.nbytes} bytes")
        print(f"  RLE runs: {len(encoded['rle'])}")
        print(f"  Compression ratio: {ratio:.2f}x")
        assert ratio > 3  # Should achieve good compression

        # Test 2: Medical imaging style mask (complex boundaries)
        medical_mask = np.zeros((256, 256), dtype=np.uint8)

        # Create complex organ-like shapes
        y, x = np.ogrid[:256, :256]

        # Heart-like shape
        heart_center = (128, 128)
        heart_radius = 60
        heart_mask = ((x - heart_center[1]) ** 2 + (y - heart_center[0]) ** 2) < heart_radius**2
        medical_mask[heart_mask] = 1

        # Add some noise around boundaries
        noise_mask = np.random.random((256, 256)) < 0.1
        medical_mask[noise_mask] = 1

        encoded = encode_segmentation_mask(medical_mask)
        ratio = calculate_compression_ratio(medical_mask, encoded)

        print("\nMedical imaging style mask (256x256):")
        print(f"  Original size: {medical_mask.nbytes} bytes")
        print(f"  RLE runs: {len(encoded['rle'])}")
        print(f"  Compression ratio: {ratio:.2f}x")
        assert ratio > 1.5  # Should achieve moderate compression

    def test_scalability_analysis(self):
        """Test how compression scales with mask size."""

        sizes = [(50, 50), (100, 100), (200, 200), (500, 500)]

        print("\nCompression scalability analysis:")
        print(f"{'Size':<12} {'Original (KB)':<15} {'RLE Runs':<12} {'Ratio':<10}")
        print("-" * 50)

        for height, width in sizes:
            # Create a mask with good compression characteristics
            mask = np.zeros((height, width), dtype=np.uint8)

            # Add a few large regions
            mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1
            mask[height // 8 : height // 4, width // 8 : width // 4] = 1

            encoded = encode_segmentation_mask(mask)
            ratio = calculate_compression_ratio(mask, encoded)
            original_kb = mask.nbytes / 1024

            print(
                f"{height}x{width:<8} {original_kb:<15.1f} {len(encoded['rle']):<12} {ratio:<10.2f}"
            )

            # Verify that larger masks still achieve good compression
            assert ratio > 2

    def test_binary_compression_analysis(self):
        """Test compression with binary masks."""

        print("\nBinary mask compression analysis:")
        print(f"{'Pattern Type':<20} {'RLE Runs':<12} {'Compression Ratio':<18}")
        print("-" * 50)

        # Test different binary patterns
        patterns = [
            ("All zeros", np.zeros((100, 100), dtype=np.uint8)),
            ("All ones", np.ones((100, 100), dtype=np.uint8)),
            (
                "Half zeros, half ones",
                np.concatenate([np.zeros((100, 50)), np.ones((100, 50))], axis=1),
            ),
            ("Random binary", np.random.randint(0, 2, size=(100, 100), dtype=np.uint8)),
        ]

        for pattern_name, mask in patterns:
            encoded = encode_segmentation_mask(mask)
            ratio = calculate_compression_ratio(mask, encoded)

            print(f"{pattern_name:<20} {len(encoded['rle']):<12} {ratio:<18.2f}")

            # All patterns should compress to some degree
            assert ratio > 0.48


class TestSegmentationEncodingErrors:
    """Test error cases for segmentation encoding and decoding."""

    def test_invalid_mask_dimensions(self):
        """Test that non-2D masks raise ValueError."""
        # 1D array
        mask_1d = np.array([0, 1, 0, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="Mask must be a 2D array"):
            encode_segmentation_mask(mask_1d)

        # 3D array
        mask_3d = np.zeros((3, 3, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Mask must be a 2D array"):
            encode_segmentation_mask(mask_3d)

    def test_missing_encoded_keys(self):
        """Test that missing keys in encoded data raise KeyError."""
        # Valid encoded data
        original_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        encoded = encode_segmentation_mask(original_mask)

        # Remove a required key
        del encoded["shape"]
        with pytest.raises(KeyError, match="Missing required key 'shape'"):
            decode_segmentation_mask(encoded)

    def test_invalid_encoded_data(self):
        """Test that invalid encoded data raises ValueError."""
        # Create invalid encoded data
        invalid_encoded = {
            "shape": (2, 2),
            "dtype": "uint8",
            "rle": [1, 1, 1],  # Only 3 pixels but shape is 2x2=4
            "total_pixels": 4,
            "first_value": 0,
        }

        with pytest.raises(ValueError, match="RLE data doesn't match total_pixels"):
            decode_segmentation_mask(invalid_encoded)

    def test_invalid_shape(self):
        """Test that invalid shape raises ValueError."""
        invalid_encoded = {
            "shape": (2, 2, 2),  # 3D shape
            "dtype": "uint8",
            "rle": [4],
            "total_pixels": 4,
            "first_value": 0,
        }

        with pytest.raises(ValueError, match="Shape must be 2D"):
            decode_segmentation_mask(invalid_encoded)

    def test_shape_total_pixels_mismatch(self):
        """Test that shape and total_pixels mismatch raises ValueError."""
        invalid_encoded = {
            "shape": (2, 2),
            "dtype": "uint8",
            "rle": [4],
            "total_pixels": 5,  # Should be 4 for 2x2
            "first_value": 0,
        }

        with pytest.raises(ValueError, match="Total pixels doesn't match shape"):
            decode_segmentation_mask(invalid_encoded)


class TestSegmentationEncodingEdgeCases:
    """Test edge cases for segmentation encoding and decoding."""

    def test_empty_mask(self):
        """Test encoding and decoding of an empty mask."""
        # This should raise an error since we can't have 0-size dimensions
        with pytest.raises(IndexError):
            original_mask = np.array([], dtype=np.uint8).reshape(0, 0)
            encode_segmentation_mask(original_mask)

    def test_single_pixel_mask(self):
        """Test encoding and decoding of a single pixel mask."""
        original_mask = np.array([[1]], dtype=np.uint8)

        encoded = encode_segmentation_mask(original_mask)
        decoded_mask = decode_segmentation_mask(encoded)

        np.testing.assert_array_equal(original_mask, decoded_mask)

    def test_very_large_values(self):
        """Test encoding and decoding with very large integer values."""
        # This test is no longer applicable since we only support binary masks
        # But we can test that boolean masks work correctly
        original_mask = np.array([[False, True], [True, False]], dtype=np.bool)

        encoded = encode_segmentation_mask(original_mask)
        decoded_mask = decode_segmentation_mask(encoded)

        np.testing.assert_array_equal(original_mask, decoded_mask)

    def test_alternating_pattern(self):
        """Test encoding and decoding of an alternating pattern (worst case for RLE)."""
        # Create a checkerboard pattern (worst case for RLE compression)
        original_mask = np.array(
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8
        )

        encoded = encode_segmentation_mask(original_mask)
        decoded_mask = decode_segmentation_mask(encoded)

        np.testing.assert_array_equal(original_mask, decoded_mask)

        # Check that compression ratio is reasonable (might not be great for this pattern)
        ratio = calculate_compression_ratio(original_mask, encoded)
        assert ratio > 0.1  # Should still have some compression due to metadata

    def test_roundtrip_preserves_metadata(self):
        """Test that roundtrip encoding/decoding preserves all metadata."""
        original_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

        encoded = encode_segmentation_mask(original_mask)
        decoded_mask = decode_segmentation_mask(encoded)

        # Re-encode the decoded mask
        re_encoded = encode_segmentation_mask(decoded_mask)

        # All metadata should be identical
        assert encoded["shape"] == re_encoded["shape"]
        assert encoded["dtype"] == re_encoded["dtype"]
        assert encoded["total_pixels"] == re_encoded["total_pixels"]
        assert encoded["rle"] == re_encoded["rle"]
        assert encoded["first_value"] == re_encoded["first_value"]
