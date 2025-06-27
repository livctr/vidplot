from typing import Any, Dict
from . import DataStreamer, KnownDurationProtocol

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
        if streamer.duration <= 0:
            raise ValueError("streamer.duration must be a positive number.")
        self._streamer = streamer

    @property
    def duration(self) -> float:
        return self._streamer.duration

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
