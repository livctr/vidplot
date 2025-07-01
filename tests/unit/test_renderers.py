import numpy as np
import pytest
from vidplot.renderers import (
    BoxRenderer,
    COCOKeypointsRenderer,
    COCOKeypoints3DRenderer,
    ProgressRenderer,
    HorizontalLabelBarRenderer,
)
from vidplot.core import StaticDataStreamer
import cv2

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
        grid_row=(1, 1),
        grid_column=(1, 1),
    )
    bbox = (0, 0, CANVAS_SHAPE[1], CANVAS_SHAPE[0])
    out = r.render(data, bbox, canvas.copy())
    cv2.imwrite("tests/output/renderers/box_renderer.png", out)
    assert out.shape == CANVAS_SHAPE
    assert out.dtype == np.uint8


def test_coco_keypoints_renderer(canvas):
    pose = np.array([[0.2, 0.2, 1.0], [0.8, 0.2, 0.8], [0.5, 0.8, 0.9]])
    r = COCOKeypointsRenderer(
        name="kp",
        data_streamer=StaticDataStreamer("static", pose),
        grid_row=(1, 1),
        grid_column=(1, 1),
    )
    bbox = (0, 0, CANVAS_SHAPE[1], CANVAS_SHAPE[0])
    out = r.render(pose, bbox, canvas.copy())
    cv2.imwrite("tests/output/renderers/coco_keypoints_renderer.png", out)
    assert out.shape == CANVAS_SHAPE
    assert out.dtype == np.uint8


def test_coco_keypoints3d_renderer(canvas):
    pose3d = np.array([[0.2, 0.2, 0.1, 1.0], [0.8, 0.2, 0.2, 0.8], [0.5, 0.8, 0.3, 0.9]])
    r = COCOKeypoints3DRenderer(
        name="kp3d",
        data_streamer=StaticDataStreamer("static", pose3d),
        grid_row=(1, 1),
        grid_column=(1, 1),
    )
    bbox = (0, 0, CANVAS_SHAPE[1], CANVAS_SHAPE[0])
    out = r.render(pose3d, bbox, canvas.copy())
    cv2.imwrite("tests/output/renderers/coco_keypoints3d_renderer.png", out)
    assert out.shape == CANVAS_SHAPE
    assert out.dtype == np.uint8


def test_progress_renderer(canvas):
    r = ProgressRenderer(
        name="progress",
        data_streamer=StaticDataStreamer("static", 0.5),
        grid_row=(1, 1),
        grid_column=(1, 1),
    )
    bbox = (0, 0, CANVAS_SHAPE[1], CANVAS_SHAPE[0])
    out = r.render(0.5, bbox, canvas.copy())
    cv2.imwrite("tests/output/renderers/progress_renderer.png", out)
    # Check that the bar area is not white (should be colored)
    y, x = CANVAS_SHAPE[0] // 2, CANVAS_SHAPE[1] // 2
    assert not np.all(out[y, x] == 255)
    assert out.shape == CANVAS_SHAPE
    assert out.dtype == np.uint8


def test_horizontal_label_bar_renderer(canvas):
    labels = ["A", "B", "A", "C", "B", "A"]
    r = HorizontalLabelBarRenderer(
        name="bar",
        data_streamer=StaticDataStreamer("static", labels),
        label_to_color={"A": (255, 0, 0), "B": (0, 255, 0), "C": (0, 0, 255)},
        grid_row=(1, 1),
        grid_column=(1, 1),
        height=20,
    )
    bbox = (0, 0, CANVAS_SHAPE[1], 20)
    out = r.render(labels, bbox, canvas.copy())
    cv2.imwrite("tests/output/renderers/horizontal_label_bar_renderer.png", out)
    # Check that the bar area is not white
    y, x = 10, CANVAS_SHAPE[1] // 2
    assert not np.all(out[y, x] == 255)
    assert out.shape == CANVAS_SHAPE
    assert out.dtype == np.uint8
