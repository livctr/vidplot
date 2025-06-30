import numpy as np
from typing import Any, Dict, Optional, Tuple
import cv2
from vidplot.core import DataStreamer, SizedStreamerProtocol
from .utils import _stream_with_last_frame_handling


class OpenCVVideoStreamer(DataStreamer, SizedStreamerProtocol):
    def __init__(self, name: str, path: str, sample_rate: float = 30.0) -> None:
        super().__init__(name, sample_rate)

        # open video
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video {path!r}")

        # read container info
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            raise ValueError("Could not read FPS from video")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # compute duration once
        self._duration = self.total_frames / self.fps

        # capture size
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._size = (width, height)

        # for nearest-frame logic
        self._prev_ts = None
        self._prev_frame = None
        self._cur_ts = None
        self._cur_frame = None
        self._last_frame_time = None
        self._last_frame = None

    @property
    def approx_duration(self) -> float:
        return self._duration

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    def _seek(self) -> Tuple[float, np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration

        # NOTE: Not exactly accurate, but hopefully monotonic
        ts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        return ts, frame

    def stream(self) -> Any:
        """
        Grab frames until we reach or pass self._clock (in seconds),
        then return whichever of the two surrounding frames is closest.
        Raises StopIteration when we've processed the last frame at target sample rate.
        """
        target_time = self._clock

        (
            frame,
            self._prev_ts,
            self._prev_frame,
            self._cur_ts,
            self._cur_frame,
            self._last_frame_time,
            self._last_frame,
        ) = _stream_with_last_frame_handling(
            target_time,
            self._prev_ts,
            self._prev_frame,
            self._cur_ts,
            self._cur_frame,
            self._last_frame_time,
            self._last_frame,
            self.sample_rate,
            self._seek,
            "nearest",
        )

        return frame


class PyAVVideoStreamer(DataStreamer, SizedStreamerProtocol):
    def __init__(self, name: str, path: str, sample_rate: float = 30.0) -> None:
        import av

        super().__init__(name, sample_rate)
        self.container = av.open(path)
        self.stream_vid = self.container.streams.video[0]
        self.stream_vid.thread_type = "AUTO"

        # Pre-iterator for decoded frames
        self.frame_iter = self.container.decode(video=0)

        # duration and size
        self._duration = float(self.stream_vid.duration * self.stream_vid.time_base)
        self._size = (self.stream_vid.width, self.stream_vid.height)

        self._prev_ts: Optional[float] = None
        self._prev_frame: Any = None
        self._cur_ts: Optional[float] = None
        self._cur_frame: Any = None
        self._last_frame_time: Optional[float] = None
        self._last_frame: Any = None

    @property
    def approx_duration(self) -> float:
        return self._duration

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    def _seek(self) -> Tuple[float, Any]:
        frame = next(self.frame_iter)
        ts = float(frame.pts * frame.time_base)
        img = frame.to_ndarray(format="bgr24")
        return ts, img

    def stream(self) -> Any:
        target = self._clock

        (
            frame,
            self._prev_ts,
            self._prev_frame,
            self._cur_ts,
            self._cur_frame,
            self._last_frame_time,
            self._last_frame,
        ) = _stream_with_last_frame_handling(
            target,
            self._prev_ts,
            self._prev_frame,
            self._cur_ts,
            self._cur_frame,
            self._last_frame_time,
            self._last_frame,
            self.sample_rate,
            self._seek,
            "nearest",
        )

        return frame


class DecordVideoStreamer(DataStreamer, SizedStreamerProtocol):
    def __init__(self, name: str, path: str, sample_rate: float = 30.0) -> None:
        from decord import VideoReader, cpu

        super().__init__(name, sample_rate)
        self.vr = VideoReader(path, ctx=cpu(0))
        self.fps = float(self.vr.get_avg_fps())
        self._duration = len(self.vr) / self.fps
        self._size = int(self.vr[0].shape[1]), int(self.vr[0].shape[0])

        self._idx = 0
        self._prev_ts: Optional[float] = None
        self._prev_frame: Any = None
        self._cur_ts: Optional[float] = None
        self._cur_frame: Any = None
        self._last_frame_time: Optional[float] = None
        self._last_frame: Any = None

    @property
    def approx_duration(self) -> float:
        return self._duration

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    def _seek(self) -> Tuple[float, Any]:
        if self._idx >= len(self.vr):
            raise StopIteration
        frame = self.vr[self._idx]
        ts = self._idx / self.fps
        self._idx += 1
        nd = frame.asnumpy() if hasattr(frame, "asnumpy") else frame
        return ts, nd

    def stream(self) -> Any:
        target = self._clock

        (
            frame,
            self._prev_ts,
            self._prev_frame,
            self._cur_ts,
            self._cur_frame,
            self._last_frame_time,
            self._last_frame,
        ) = _stream_with_last_frame_handling(
            target,
            self._prev_ts,
            self._prev_frame,
            self._cur_ts,
            self._cur_frame,
            self._last_frame_time,
            self._last_frame,
            self.sample_rate,
            self._seek,
            "nearest",
        )

        return frame


class VideoStreamer(DataStreamer, SizedStreamerProtocol):
    """
    Factory wrapper that selects backend among 'opencv', 'pyav', or 'decord'.
    Delegates DataStreamer interface to chosen implementation.
    """

    def __init__(
        self, name: str, path: str, backend: str = "opencv", sample_rate: float = 30.0
    ) -> None:
        super().__init__(name, sample_rate)
        backend = backend.lower()
        if backend == "opencv":
            self._impl = OpenCVVideoStreamer(name, path, sample_rate)
        elif backend == "pyav":
            self._impl = PyAVVideoStreamer(name, path, sample_rate)
        elif backend == "decord":
            self._impl = DecordVideoStreamer(name, path, sample_rate)
        else:
            raise ValueError(f"Unknown backend {backend}")

    @property
    def approx_duration(self) -> float:
        return self._impl.approx_duration

    @property
    def size(self) -> Tuple[int, int]:
        return self._impl.size

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._impl.metadata

    def stream(self) -> Any:
        return self._impl.stream()

    def __iter__(self):
        return iter(self._impl)

    def __next__(self):
        return next(self._impl)
