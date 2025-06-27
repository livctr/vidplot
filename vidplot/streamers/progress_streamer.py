from typing import Any, Dict
from vidplot.core import DataStreamer, KnownDurationProtocol


class ProgressStreamer(DataStreamer, KnownDurationProtocol):
    """
    A progress streamer that returns the current fraction of time passed in the stream.
    Wraps another streamer with known duration and provides progress tracking.
    """

    def __init__(
        self,
        name: str,
        streamer: KnownDurationProtocol,
        sample_rate: float = 30.0,
    ):
        super().__init__(name=name, sample_rate=sample_rate)
        if streamer is None:
            raise ValueError("streamer cannot be None")
        if not hasattr(streamer, "duration"):
            raise ValueError(
                "Progress streamer must track progress from a streamer of known duration."
            )
        if streamer.duration <= 0:
            raise ValueError("streamer.duration must be a positive number.")

        self._streamer = streamer

    @property
    def duration(self) -> float:
        """This duration is to tell other streamers the exact duration."""
        return self._streamer.duration

    @property
    def approx_duration(self) -> float:
        """
        This duration is for the tqdm bar. For video streamers, duration is
        not 100% accurate, which is why this is implemented. However, for a streamer
        of KnownDurationProtocol, the two are the same.
        """
        return self._streamer.approx_duration

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "wrapped_streamer": self._streamer.name,
            "progress_tracker": True,
        }

    def stream(self) -> float:
        """
        Returns the current progress as a fraction (0.0 to 1.0).

        Returns:
            Current progress as a float between 0.0 and 1.0.
        """
        if self._clock >= self.duration:
            raise StopIteration("Reached end of progress stream")
        return self._clock / self.duration
