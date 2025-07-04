import numpy as np
import pytest
import os
from vidplot.renderers import (
    BoxRenderer,
    COCOKeypointsRenderer,
    COCOKeypoints3DRenderer,
    LabelBarRenderer,
    RGBRenderer,
    StringRenderer,
)
from vidplot.core import StaticDataStreamer
from vidplot.streamers import LabelBarStreamer, VideoStreamer
import cv2
from vidplot.core.video_canvas import VideoCanvas

CANVAS_SHAPE = (240, 320, 3)


@pytest.fixture
def canvas():
    return np.ones(CANVAS_SHAPE, dtype=np.uint8) * 255


def test_box_renderer(canvas):
    data = {
        "shape": (CANVAS_SHAPE[0], CANVAS_SHAPE[1]),
        "boxes": [
            {"box": [10, 10, 50, 50], "text_label": "A", "score": 0.9, "id": 1},
            {"box": [60, 60, 90, 90], "id": 2, "score": 0.7},
        ],
    }
    r = BoxRenderer(
        name="box",
        data_streamer=StaticDataStreamer("static", data),
        id_to_color={1: (255, 0, 0), 2: (0, 255, 0)},
    )
    vc = VideoCanvas(row_gap=0, col_gap=0)
    vc.attach(
        r.data_streamer,
        r,
        grid_row=1,
        grid_col=1,
        height=[CANVAS_SHAPE[0]],
        width=[CANVAS_SHAPE[1]],
    )
    output_path = "tests/output/renderers/box_renderer.png"
    vc.write(output_path)
    assert os.path.exists(output_path)


def test_coco_keypoints_renderer(canvas):
    pose = np.array([[0.2, 0.2, 1.0], [0.8, 0.2, 0.8], [0.5, 0.8, 0.9]])
    r = COCOKeypointsRenderer(
        name="kp",
        data_streamer=StaticDataStreamer("static", pose),
    )
    vc = VideoCanvas(row_gap=0, col_gap=0)
    vc.attach(
        r.data_streamer,
        r,
        grid_row=1,
        grid_col=1,
        height=[CANVAS_SHAPE[0]],
        width=[CANVAS_SHAPE[1]],
    )
    output_path = "tests/output/renderers/coco_keypoints_renderer.png"
    vc.write(output_path)
    assert os.path.exists(output_path)


def test_coco_keypoints3d_renderer(canvas):
    pose3d = np.array([[0.2, 0.2, 0.1, 1.0], [0.8, 0.2, 0.2, 0.8], [0.5, 0.8, 0.3, 0.9]])
    r = COCOKeypoints3DRenderer(
        name="kp3d",
        data_streamer=StaticDataStreamer("static", pose3d),
    )
    vc = VideoCanvas(row_gap=0, col_gap=0)
    vc.attach(
        r.data_streamer,
        r,
        grid_row=1,
        grid_col=1,
        height=[CANVAS_SHAPE[0]],
        width=[CANVAS_SHAPE[1]],
    )
    output_path = "tests/output/renderers/coco_keypoints3d_renderer.png"
    vc.write(output_path)
    assert os.path.exists(output_path)


def test_label_bar_renderer(canvas):
    # Create a LabelBarStreamer with proper data
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    labels = ["A", "B", "A", "C", "B", "A"]
    r = LabelBarRenderer(
        name="bar",
        data_streamer=LabelBarStreamer("label_stream", time=times, data=labels, duration=5.0),
        label_to_color={"A": (255, 0, 0), "B": (0, 255, 0), "C": (0, 0, 255)},
        height=20,
    )
    vc = VideoCanvas(row_gap=0, col_gap=0)
    vc.attach(r.data_streamer, r, grid_row=1, grid_col=1, height=[20], width=[CANVAS_SHAPE[1]])
    output_path = "tests/output/renderers/label_bar_renderer.png"
    vc.write(output_path)
    assert os.path.exists(output_path)


def test_rgb_renderer(canvas):
    # Create a dummy video file for testing
    dummy_video_path = "tests/output/renderers/dummy_video.mp4"
    os.makedirs(os.path.dirname(dummy_video_path), exist_ok=True)

    # Create a simple video file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dummy_video_path, fourcc, 30, (CANVAS_SHAPE[1], CANVAS_SHAPE[0]))
    frame = np.ones((CANVAS_SHAPE[0], CANVAS_SHAPE[1], 3), dtype=np.uint8) * 127
    writer.write(frame)
    writer.release()

    r = RGBRenderer(
        name="rgb",
        data_streamer=VideoStreamer("video", dummy_video_path),
    )
    vc = VideoCanvas(row_gap=0, col_gap=0)
    vc.attach(
        r.data_streamer,
        r,
        grid_row=1,
        grid_col=1,
        height=[CANVAS_SHAPE[0]],
        width=[CANVAS_SHAPE[1]],
    )
    output_path = "tests/output/renderers/rgb_renderer.png"
    vc.write(output_path)
    assert os.path.exists(output_path)


def test_string_renderer_attach(canvas):
    # Test attach: StringRenderer overlays on top of RGBRenderer
    # Create a dummy video file for testing
    dummy_video_path = "tests/output/renderers/dummy_video_string.mp4"
    os.makedirs(os.path.dirname(dummy_video_path), exist_ok=True)

    # Create a simple video file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dummy_video_path, fourcc, 30, (CANVAS_SHAPE[1], CANVAS_SHAPE[0]))
    frame = np.ones((CANVAS_SHAPE[0], CANVAS_SHAPE[1], 3), dtype=np.uint8) * 200
    writer.write(frame)
    writer.release()

    rgb = RGBRenderer(
        name="rgb",
        data_streamer=VideoStreamer("video", dummy_video_path),
    )
    text = "Overlay!"
    string = StringRenderer(
        name="string",
        data_streamer=StaticDataStreamer("static", text),
    )
    # Attach string renderer as a child to rgb renderer
    # Use VideoCanvas for placement
    vc = VideoCanvas(row_gap=0, col_gap=0)
    vc.attach(
        rgb.data_streamer,
        rgb,
        grid_row=1,
        grid_col=1,
        height=[CANVAS_SHAPE[0]],
        width=[CANVAS_SHAPE[1]],
    )
    vc.attach(
        string.data_streamer,
        string,
        grid_row=1,
        grid_col=1,
        height=[CANVAS_SHAPE[0]],
        width=[CANVAS_SHAPE[1]],
    )
    output_path = "tests/output/renderers/rgb_with_string_attach.png"
    vc.write(output_path)
    assert os.path.exists(output_path)
