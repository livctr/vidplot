import os
import numpy as np
import cv2
import pytest
from pathlib import Path
from vidplot.core.streamer import DataStreamer
from vidplot.core.renderer import Renderer
from vidplot.core.orchestrator import AnnotationOrchestrator


class MovingSquareStreamer(DataStreamer):
    def __init__(
        self,
        name,
        sample_rate=30.0,
        duration=5.0,
        frame_size=(64, 64),
        square_size=16,
    ):
        super().__init__(name=name, sample_rate=sample_rate)
        self.duration = duration
        self.frame_size = frame_size
        self.square_size = square_size
        self._n_frames = int(duration * sample_rate)
        self._frame_idx = 0

    @property
    def approx_duration(self):
        return self.duration

    def stream(self):
        if self._frame_idx >= self._n_frames:
            raise StopIteration
        # Move square horizontally
        img = np.ones((*self.frame_size, 3), dtype=np.uint8) * 255
        t = self._frame_idx / self._n_frames
        x = int(t * (self.frame_size[1] - self.square_size))
        y = (self.frame_size[0] - self.square_size) // 2
        img[y : y + self.square_size, x : x + self.square_size] = 0
        self._frame_idx += 1
        return img


class SimpleRGBRenderer(Renderer):
    def _default_size(self):
        return (64, 64)

    def _render(self, data, bbox, canvas):
        x1, y1, x2, y2 = bbox
        if data is not None:
            frame = cv2.resize(data, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
            canvas[y1:y2, x1:x2] = frame
        return canvas


class ColoredSquareRenderer(Renderer):
    def __init__(
        self,
        name,
        data_streamer,
        grid_row,
        grid_column,
        color=(255, 0, 0),
        z_index=0,
    ):
        super().__init__(name, data_streamer, grid_row, grid_column, z_index=z_index)
        self.color = color

    def _default_size(self):
        return (64, 64)

    def _render(self, data, bbox, canvas):
        x1, y1, x2, y2 = bbox
        # Draw a colored square
        cv2.rectangle(canvas, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), self.color, -1)
        return canvas


def test_cell_coordinates_and_canvas_shape():
    """Test that cell coordinates and canvas shape are computed correctly."""
    os.makedirs(output_dir, exist_ok=True)

    # Test with simple 2x2 grid, no gaps
    orch = AnnotationOrchestrator([32, 32], [32, 32], gap=0)

    # Check canvas shape: should be (64, 64, 3) for 2x32 rows, 2x32 cols
    assert orch._canvas_shape == (64, 64, 3)

    # Check cell coordinates
    assert orch._cell_coords[(1, 1)] == (0, 0, 32, 32)  # top-left
    assert orch._cell_coords[(1, 2)] == (32, 0, 64, 32)  # top-right
    assert orch._cell_coords[(2, 1)] == (0, 32, 32, 64)  # bottom-left
    assert orch._cell_coords[(2, 2)] == (32, 32, 64, 64)  # bottom-right

    # Test with gaps
    orch_with_gaps = AnnotationOrchestrator([32, 32], [32, 32], gap=5)
    # Canvas should be (69, 69, 3): 32+32+5 gap, 32+32+5 gap
    assert orch_with_gaps._canvas_shape == (69, 69, 3)

    # Test cell coordinates with gaps
    assert orch_with_gaps._cell_coords[(1, 1)] == (0, 0, 32, 32)
    assert orch_with_gaps._cell_coords[(1, 2)] == (37, 0, 69, 32)  # 32+5 gap
    assert orch_with_gaps._cell_coords[(2, 1)] == (0, 37, 32, 69)  # 32+5 gap


output_dir = Path("tests/output/orchestrator")


def test_video_30fps():
    os.makedirs(output_dir, exist_ok=True)
    streamer = MovingSquareStreamer("sq", sample_rate=30.0)
    renderer = SimpleRGBRenderer("r", streamer, (1, 2), (1, 2))
    orch = AnnotationOrchestrator([32, 32], [32, 32], gap=5)
    orch.set_annotators([streamer], [renderer], [("sq", "r")])
    outpath = str(output_dir / "video_30fps.mp4")
    orch.show_layout(str(output_dir / "layout_30fps.png"))
    orch.write(outpath, fps=30.0)
    assert os.path.exists(outpath)


def test_video_1fps():
    os.makedirs(output_dir, exist_ok=True)
    streamer = MovingSquareStreamer("sq", sample_rate=30.0)
    renderer = SimpleRGBRenderer("r", streamer, (1, 1), (1, 1))
    orch = AnnotationOrchestrator([64], [64], gap=5)
    orch.set_annotators([streamer], [renderer], [("sq", "r")])
    outpath = str(output_dir / "video_1fps.mp4")
    orch.show_layout(str(output_dir / "layout_1fps.png"))
    orch.write(outpath, fps=1.0)
    assert os.path.exists(outpath)


def test_video_side_by_side_diff_fps():
    os.makedirs(output_dir, exist_ok=True)
    streamer1 = MovingSquareStreamer("sq1", sample_rate=30.0)
    streamer2 = MovingSquareStreamer("sq2", sample_rate=1.0)
    renderer1 = SimpleRGBRenderer("r1", streamer1, (1, 1), (1, 1))
    renderer2 = SimpleRGBRenderer("r2", streamer2, (1, 1), (2, 2))
    orch = AnnotationOrchestrator([64], [64, 64], gap=5)
    orch.set_annotators(
        [streamer1, streamer2],
        [renderer1, renderer2],
        [("sq1", "r1"), ("sq2", "r2")],
    )
    outpath = str(output_dir / "video_side_by_side_diff_fps.mp4")
    orch.show_layout(str(output_dir / "layout_side_by_side_diff_fps.png"))
    orch.write(outpath, fps=30.0)
    assert os.path.exists(outpath)


def test_z_index_functionality():
    """Test that renderers with higher z_index are rendered on top."""
    os.makedirs(output_dir, exist_ok=True)

    # Create two renderers that overlap in the same cell
    # Red square (z_index=0) should be underneath
    # Blue square (z_index=1) should be on top
    streamer1 = MovingSquareStreamer("sq1", sample_rate=30.0)
    streamer2 = MovingSquareStreamer("sq2", sample_rate=30.0)

    red_renderer = ColoredSquareRenderer(
        "red", streamer1, (1, 1), (1, 1), color=(0, 0, 255), z_index=0
    )
    blue_renderer = ColoredSquareRenderer(
        "blue", streamer2, (1, 1), (1, 1), color=(255, 0, 0), z_index=1
    )

    orch = AnnotationOrchestrator([64], [64])
    orch.set_annotators(
        [streamer1, streamer2],
        [red_renderer, blue_renderer],
        [("sq1", "red"), ("sq2", "blue")],
    )

    outpath = str(output_dir / "video_z_index_test.mp4")
    orch.show_layout(str(output_dir / "layout_z_index_test.png"))
    orch.write(outpath, fps=30.0)
    assert os.path.exists(outpath)

    # Verify that routes are sorted by z_index (lower first)
    assert orch.routes[0][1] == "red"  # z_index=0 should come first
    assert orch.routes[1][1] == "blue"  # z_index=1 should come second


def test_different_duration_streamers():
    """Test that orchestrator handles streamers with different durations correctly."""
    os.makedirs(output_dir, exist_ok=True)

    # Create two streamers with different durations
    short_streamer = MovingSquareStreamer("short", sample_rate=30.0, duration=2.0)  # 2 seconds
    long_streamer = MovingSquareStreamer("long", sample_rate=30.0, duration=5.0)  # 5 seconds

    renderer1 = SimpleRGBRenderer("r1", short_streamer, (1, 1), (1, 1))
    renderer2 = SimpleRGBRenderer("r2", long_streamer, (1, 1), (2, 2))

    orch = AnnotationOrchestrator([64], [64, 64])
    orch.set_annotators(
        [short_streamer, long_streamer],
        [renderer1, renderer2],
        [("short", "r1"), ("long", "r2")],
    )

    outpath = str(output_dir / "video_different_durations.mp4")
    orch.show_layout(str(output_dir / "layout_different_durations.png"))
    orch.write(outpath, fps=30.0)

    assert os.path.exists(outpath)

    # Verify that the video stops when the shorter streamer ends
    # The orchestrator should stop at 2 seconds (shortest duration)
    # This is handled by the while loop breaking when any streamer is done


def test_renderer_out_of_bounds():
    """Test that orchestrator raises error when renderer is out of grid bounds."""
    os.makedirs(output_dir, exist_ok=True)

    streamer = MovingSquareStreamer("sq", sample_rate=30.0)

    # Test row out of bounds (grid has 1 row, renderer tries to use row 2)
    renderer_row_oob = SimpleRGBRenderer("r1", streamer, (2, 2), (1, 1))
    orch_row = AnnotationOrchestrator([64], [64])

    with pytest.raises(ValueError, match="Renderer r1 row out of bounds"):
        orch_row.set_annotators([streamer], [renderer_row_oob], [("sq", "r1")])

    # Test column out of bounds (grid has 1 column, renderer tries to use column 2)
    renderer_col_oob = SimpleRGBRenderer("r2", streamer, (1, 1), (2, 2))
    orch_col = AnnotationOrchestrator([64], [64])

    with pytest.raises(ValueError, match="Renderer r2 col out of bounds"):
        orch_col.set_annotators([streamer], [renderer_col_oob], [("sq", "r2")])

    # Test both row and column out of bounds
    renderer_both_oob = SimpleRGBRenderer("r3", streamer, (2, 2), (2, 2))
    orch_both = AnnotationOrchestrator([64], [64])

    with pytest.raises(ValueError, match="Renderer r3 row out of bounds"):
        orch_both.set_annotators([streamer], [renderer_both_oob], [("sq", "r3")])


def test_valid_renderer_bounds():
    """Test that valid renderer bounds work correctly."""
    os.makedirs(output_dir, exist_ok=True)

    streamer = MovingSquareStreamer("sq", sample_rate=30.0)

    # Test single cell renderer
    renderer_single = SimpleRGBRenderer("r1", streamer, (1, 1), (1, 1))
    orch_single = AnnotationOrchestrator([64], [64])
    orch_single.set_annotators([streamer], [renderer_single], [("sq", "r1")])

    # Test multi-cell renderer (spans 2x2 grid)
    renderer_multi = SimpleRGBRenderer("r2", streamer, (1, 2), (1, 2))
    orch_multi = AnnotationOrchestrator([32, 32], [32, 32])
    orch_multi.set_annotators([streamer], [renderer_multi], [("sq", "r2")])

    # Verify cell coordinates are computed correctly for multi-cell renderer
    x1, y1 = orch_multi._cell_coords[(1, 1)][:2]  # start cell
    x2, y2 = orch_multi._cell_coords[(2, 2)][2:]  # end cell
    assert x1 == 0 and y1 == 0
    assert x2 == 64 and y2 == 64  # 32+32 with no gap


def test_empty_streamers():
    """Test that orchestrator handles empty streamer list correctly."""
    orch = AnnotationOrchestrator([64], [64])

    # Should not raise error with empty lists
    orch.set_annotators([], [], [])
    assert len(orch.streamers) == 0
    assert len(orch.renderers) == 0
    assert len(orch.routes) == 0


def test_duplicate_names():
    """Test that orchestrator handles duplicate names correctly."""
    streamer1 = MovingSquareStreamer("sq", sample_rate=30.0)
    streamer2 = MovingSquareStreamer("sq", sample_rate=30.0)  # Same name
    renderer1 = SimpleRGBRenderer("r", streamer1, (1, 1), (1, 1))
    renderer2 = SimpleRGBRenderer("r", streamer2, (1, 1), (1, 1))  # Same name

    orch = AnnotationOrchestrator([64], [64])

    # This should work (later items overwrite earlier ones)
    orch.set_annotators(
        [streamer1, streamer2],
        [renderer1, renderer2],
        [("sq", "r"), ("sq", "r")],
    )

    # Should only have one streamer and one renderer (last one wins)
    assert len(orch.streamers) == 1
    assert len(orch.renderers) == 1
    assert len(orch.routes) == 2  # But routes are preserved


if __name__ == "__main__":
    import sys

    test_functions = [
        test_cell_coordinates_and_canvas_shape,
        test_video_30fps,
        test_video_1fps,
        test_video_side_by_side_diff_fps,
        test_z_index_functionality,
        test_different_duration_streamers,
        test_renderer_out_of_bounds,
        test_valid_renderer_bounds,
        test_empty_streamers,
        test_duplicate_names,
    ]

    for test_func in test_functions:
        print(f"Running {test_func.__name__} ...", flush=True)
        try:
            test_func()
            print(f"  {test_func.__name__} PASSED", flush=True)
        except Exception as e:
            print(f"  {test_func.__name__} FAILED: {e}", flush=True)
            import traceback

            traceback.print_exc()
            sys.exit(1)
    print("All tests completed.")
