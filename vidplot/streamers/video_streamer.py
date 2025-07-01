import numpy as np
from typing import Any, Optional, Tuple
import cv2
from vidplot.core import DataStreamer, SizedStreamerProtocol


class VideoStreamer(DataStreamer, SizedStreamerProtocol):
    def __init__(
        self,
        name: str,
        path: str,
        backend: str = "opencv",  # one of 'opencv', 'pyav', 'decord'
        sample_rate: float = 30.0,
        duration: Optional[float] = None,
        stream_method: str = "nearest_neighbor",
        tol: float = 1e-5,
    ) -> None:
        # Validate backend and stream method
        assert backend in ["opencv", "pyav", "decord"], f"Unsupported backend '{backend}'"
        assert stream_method in [
            "nearest_neighbor",
            "LOCF",
        ], f"Unsupported stream method '{stream_method}'"

        super().__init__(name, sample_rate)
        self.backend = backend
        self._stream_method = stream_method
        self._tol = tol
        self._seeked_num = 0
        self._buf: list[Tuple[float, np.ndarray]] = []

        # Open and inspect video based on backend
        if backend == "opencv":
            import cv2

            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video {path!r}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        elif backend == "pyav":
            import av

            container = av.open(path)
            vs = container.streams.video[0]
            self._pyav_iter = container.decode(vs)
            self.fps = float(vs.average_rate) if vs.average_rate else 0.0
            self.total_frames = vs.frames or 0
            width, height = vs.width, vs.height

        else:  # decord
            from decord import VideoReader, cpu

            self.vr = VideoReader(path, ctx=cpu(0))
            self.fps = self.vr.get_avg_fps()
            self.total_frames = len(self.vr)
            first_frame = self.vr[0]
            width, height = first_frame.shape[1], first_frame.shape[0]

        if self.fps <= 0:
            raise ValueError("Could not determine FPS for backend '{backend}'")

        # Determine duration
        if duration is not None:
            self._duration = duration
        else:
            self._duration = self.total_frames / self.fps

        self._size = (width, height)

    @property
    def approx_duration(self) -> float:
        return self._duration

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    def _seek(self) -> Tuple[float, np.ndarray]:
        # Compute timestamp
        ts = self._seeked_num / self.fps
        self._seeked_num += 1

        # Grab frame based on backend
        if self.backend == "opencv":
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                raise StopIteration
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        elif self.backend == "pyav":
            try:
                frame = next(self._pyav_iter)
            except StopIteration:
                raise
            # Convert to RGB ndarray
            frame = frame.to_ndarray(format="rgb24")

        else:  # decord
            if self._seeked_num - 1 >= self.total_frames:
                raise StopIteration
            frame = self.vr[self._seeked_num - 1]
            # decord returns RGB ndarray by default

        return ts, frame

    def stream(self) -> Any:
        """
        Grab frames until we reach or pass self._clock (in seconds),
        then return whichever of the two surrounding frames is closest.
        Raises StopIteration when we've processed the last frame at target sample rate.
        """
        target_time = self._clock

        # Grab frames until we reach or pass target_time
        while not self._buf or self._buf[-1][0] < target_time - self._tol:
            try:
                ts, payload = self._seek()
                if len(self._buf) == 2:
                    self._buf.pop(0)
                self._buf.append((ts, payload))
            except StopIteration:
                break

        # Decide which frame to return
        if self._buf and self._buf[-1][0] >= target_time - self._tol:
            # We have at least one frame past or at target
            if len(self._buf) == 1:
                return self._buf[0][1]

            # Two-frame buffer: choose between buf[0] (older) and buf[1] (newer)
            t0, f0 = self._buf[0]
            t1, f1 = self._buf[1]

            if self._stream_method == "nearest_neighbor":
                return f0 if abs(t0 - target_time) < abs(t1 - target_time) else f1

            # LOCF
            return f1 if t1 >= target_time - self._tol else f0

        if target_time <= self.approx_duration and self._buf:
            # If we are within the duration of the video, return the last frame
            return self._buf[-1][1]

        raise StopIteration  # No more frames to read
