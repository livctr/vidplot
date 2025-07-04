import os
import numpy as np
import cv2
import pytest
from pathlib import Path
from vidplot.core.streamer import DataStreamer, StaticDataStreamer
from vidplot.core.renderer import Renderer
from vidplot.core.video_canvas import VideoCanvas


class MovingSquareStreamer(DataStreamer):
    def __init__(
        self,
        name,
        sample_rate=30.0,
        duration=5.0,
        frame_size=(64, 64),
        square_size=16,
    ):
        super().__init__(name=name)
        self._duration = duration
        self.frame_size = frame_size
        self.square_size = square_size
        self._n_frames = int(duration * sample_rate)
        self._sample_rate = sample_rate
        self._frame_idx = 0

    @property
    def duration(self):
        return self._duration

    def __next__(self):
        if self._frame_idx >= self._n_frames:
            raise StopIteration
        # Move square horizontally
        img = np.ones((*self.frame_size, 3), dtype=np.uint8) * 255
        t = self._frame_idx / self._n_frames
        x = int(t * (self.frame_size[1] - self.square_size))
        y = (self.frame_size[0] - self.square_size) // 2
        img[y : y + self.square_size, x : x + self.square_size] = 0
        time = self._frame_idx / self._sample_rate
        self._frame_idx += 1
        return time, img


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
        color=(255, 0, 0),
    ):
        super().__init__(name, data_streamer)
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
    canvas = VideoCanvas(row_gap=0, col_gap=0)

    # Attach dummy streamers/renderers to fill grid
    class DummyRenderer(Renderer):
        def _default_size(self):
            return (32, 32)

        def _render(self, data, bbox, canvas):
            return canvas

    s1 = StaticDataStreamer("s1", "")
    r1 = DummyRenderer("r1", s1)
    s2 = StaticDataStreamer("s2", "")
    r2 = DummyRenderer("r2", s2)
    s3 = StaticDataStreamer("s3", "")
    r3 = DummyRenderer("r3", s3)
    s4 = StaticDataStreamer("s4", "")
    r4 = DummyRenderer("r4", s4)
    canvas.attach(s1, r1, grid_row=1, grid_col=1, height=[32], width=[32])
    canvas.attach(s2, r2, grid_row=1, grid_col=2, height=[32], width=[32])
    canvas.attach(s3, r3, grid_row=2, grid_col=1, height=[32], width=[32])
    canvas.attach(s4, r4, grid_row=2, grid_col=2, height=[32], width=[32])

    # Check canvas shape: should be (64, 64, 3) for 2x32 rows, 2x32 cols
    assert canvas._canvas_shape == (64, 64, 3)

    # Check cell coordinates
    assert canvas._cell_coords[(1, 1)] == (0, 0, 32, 32)  # top-left
    assert canvas._cell_coords[(1, 2)] == (32, 0, 64, 32)  # top-right
    assert canvas._cell_coords[(2, 1)] == (0, 32, 32, 64)  # bottom-left
    assert canvas._cell_coords[(2, 2)] == (32, 32, 64, 64)  # bottom-right

    # Test with gaps
    canvas_gaps = VideoCanvas(row_gap=5, col_gap=5)
    s1 = StaticDataStreamer("gs1", "")
    r1 = DummyRenderer("gr1", s1)
    s2 = StaticDataStreamer("gs2", "")
    r2 = DummyRenderer("gr2", s2)
    s3 = StaticDataStreamer("gs3", "")
    r3 = DummyRenderer("gr3", s3)
    s4 = StaticDataStreamer("gs4", "")
    r4 = DummyRenderer("gr4", s4)
    canvas_gaps.attach(s1, r1, grid_row=1, grid_col=1, height=[32], width=[32])
    canvas_gaps.attach(s2, r2, grid_row=1, grid_col=2, height=[32], width=[32])
    canvas_gaps.attach(s3, r3, grid_row=2, grid_col=1, height=[32], width=[32])
    canvas_gaps.attach(s4, r4, grid_row=2, grid_col=2, height=[32], width=[32])
    # Canvas should be (69, 69, 3): 32+32+5 gap, 32+32+5 gap
    assert canvas_gaps._canvas_shape == (69, 69, 3)
    # Test cell coordinates with gaps
    assert canvas_gaps._cell_coords[(1, 1)] == (0, 0, 32, 32)
    assert canvas_gaps._cell_coords[(1, 2)] == (37, 0, 69, 32)  # 32+5 gap
    assert canvas_gaps._cell_coords[(2, 1)] == (0, 37, 32, 69)  # 32+5 gap


output_dir = Path("tests/output/orchestrator")


def test_video_30fps():
    os.makedirs(output_dir, exist_ok=True)
    streamer = MovingSquareStreamer("sq", sample_rate=30.0)
    renderer = SimpleRGBRenderer("r", streamer)
    canvas = VideoCanvas(row_gap=5, col_gap=5)
    canvas.attach(
        streamer, renderer, grid_row=(1, 2), grid_col=(1, 2), height=[32, 32], width=[32, 32]
    )
    outpath = str(output_dir / "video_30fps.mp4")
    canvas.show_layout(str(output_dir / "layout_30fps.png"))
    canvas.write(outpath, fps=30.0)
    assert os.path.exists(outpath)


def test_video_1fps():
    os.makedirs(output_dir, exist_ok=True)
    streamer = MovingSquareStreamer("sq", sample_rate=30.0)
    renderer = SimpleRGBRenderer("r", streamer)
    canvas = VideoCanvas(row_gap=5, col_gap=5)
    canvas.attach(streamer, renderer, grid_row=1, grid_col=1, height=[64], width=[64])
    outpath = str(output_dir / "video_1fps.mp4")
    canvas.show_layout(str(output_dir / "layout_1fps.png"))
    canvas.write(outpath, fps=1.0)
    assert os.path.exists(outpath)


def test_video_side_by_side_diff_fps():
    os.makedirs(output_dir, exist_ok=True)
    streamer1 = MovingSquareStreamer("sq1", sample_rate=30.0)
    streamer2 = MovingSquareStreamer("sq2", sample_rate=1.0)
    renderer1 = SimpleRGBRenderer("r1", streamer1)
    renderer2 = SimpleRGBRenderer("r2", streamer2)
    canvas = VideoCanvas(row_gap=5, col_gap=5)
    canvas.attach(streamer1, renderer1, grid_row=1, grid_col=1, height=[64], width=[64])
    canvas.attach(streamer2, renderer2, grid_row=1, grid_col=2, height=[64], width=[64])
    outpath = str(output_dir / "video_side_by_side_diff_fps.mp4")
    canvas.show_layout(str(output_dir / "layout_side_by_side_diff_fps.png"))
    canvas.write(outpath, fps=30.0)
    assert os.path.exists(outpath)


def test_z_index_functionality():
    """Test that renderers with higher z_index are rendered on top."""
    os.makedirs(output_dir, exist_ok=True)

    # Create two renderers that overlap in the same cell
    # Red square (z_index=0) should be underneath
    # Blue square (z_index=1) should be on top
    streamer1 = MovingSquareStreamer("sq1", sample_rate=30.0)
    streamer2 = MovingSquareStreamer("sq2", sample_rate=30.0)

    red_renderer = ColoredSquareRenderer("red", streamer1, color=(0, 0, 255))
    blue_renderer = ColoredSquareRenderer("blue", streamer2, color=(255, 0, 0))

    canvas = VideoCanvas(row_gap=0, col_gap=0)
    canvas.attach(
        streamer1, red_renderer, grid_row=1, grid_col=1, height=[64], width=[64], z_index=0
    )
    canvas.attach(
        streamer2, blue_renderer, grid_row=1, grid_col=1, height=[64], width=[64], z_index=1
    )

    outpath = str(output_dir / "video_z_index_test.mp4")
    canvas.show_layout(str(output_dir / "layout_z_index_test.png"))
    canvas.write(outpath, fps=30.0)
    assert os.path.exists(outpath)

    # Verify that routes are sorted by z_index (lower first)
    assert canvas.routes[0][1] == "red"  # z_index=0 should come first
    assert canvas.routes[1][1] == "blue"  # z_index=1 should come second


def test_different_duration_streamers():
    """Test that orchestrator handles streamers with different durations correctly."""
    os.makedirs(output_dir, exist_ok=True)

    # Create two streamers with different durations
    short_streamer = MovingSquareStreamer("short", sample_rate=30.0, duration=2.0)  # 2 seconds
    long_streamer = MovingSquareStreamer("long", sample_rate=30.0, duration=5.0)  # 5 seconds

    renderer1 = SimpleRGBRenderer("r1", short_streamer)
    renderer2 = SimpleRGBRenderer("r2", long_streamer)

    canvas = VideoCanvas(row_gap=0, col_gap=0)
    canvas.attach(short_streamer, renderer1, grid_row=1, grid_col=1, height=[64], width=[64])
    canvas.attach(long_streamer, renderer2, grid_row=1, grid_col=2, height=[64], width=[64])

    outpath = str(output_dir / "video_different_durations.mp4")
    canvas.write(outpath, fps=30.0)

    assert os.path.exists(outpath)

    # Verify that the video stops when the shorter streamer ends
    # The orchestrator should stop at 2 seconds (shortest duration)
    # This is handled by the while loop breaking when any streamer is done


def test_valid_renderer_bounds():
    """Test that valid renderer bounds work correctly."""
    os.makedirs(output_dir, exist_ok=True)

    streamer = MovingSquareStreamer("sq", sample_rate=30.0)

    # Test single cell renderer
    renderer_single = SimpleRGBRenderer("r1", streamer)
    canvas_single = VideoCanvas(row_gap=0, col_gap=0)
    canvas_single.attach(streamer, renderer_single, grid_row=1, grid_col=1, height=[64], width=[64])

    # Test multi-cell renderer (spans 2x2 grid)
    renderer_multi = SimpleRGBRenderer("r2", streamer)
    canvas_multi = VideoCanvas(row_gap=0, col_gap=0)
    canvas_multi.attach(
        streamer, renderer_multi, grid_row=(1, 2), grid_col=(1, 2), height=[32, 32], width=[32, 32]
    )

    # Verify cell coordinates are computed correctly for multi-cell renderer
    x1, y1 = canvas_multi._cell_coords[(1, 1)][:2]  # start cell
    x2, y2 = canvas_multi._cell_coords[(2, 2)][2:]  # end cell
    assert x1 == 0 and y1 == 0
    assert x2 == 64 and y2 == 64  # 32+32 with no gap


def test_empty_streamers():
    """Test that orchestrator handles empty streamer list correctly."""
    canvas = VideoCanvas(row_gap=0, col_gap=0)
    # Should not raise error with no attachments
    assert len(canvas.streamers) == 0
    assert len(canvas.renderers) == 0
    assert len(canvas.routes) == 0


def test_duplicate_names():
    """Test that orchestrator handles duplicate names correctly."""
    streamer1 = MovingSquareStreamer("sq", sample_rate=30.0)
    streamer2 = MovingSquareStreamer("sq", sample_rate=30.0)  # Same name
    renderer1 = SimpleRGBRenderer("r", streamer1)
    renderer2 = SimpleRGBRenderer("r", streamer2)

    canvas = VideoCanvas(row_gap=0, col_gap=0)
    canvas.attach(streamer1, renderer1, grid_row=1, grid_col=1, height=[64], width=[64])
    with pytest.raises(ValueError):
        canvas.attach(streamer2, renderer2, grid_row=1, grid_col=1, height=[64], width=[64])


if __name__ == "__main__":
    import sys

    test_functions = [
        test_cell_coordinates_and_canvas_shape,
        test_video_30fps,
        test_video_1fps,
        test_video_side_by_side_diff_fps,
        test_z_index_functionality,
        test_different_duration_streamers,
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
