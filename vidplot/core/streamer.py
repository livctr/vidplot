from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Protocol, Tuple

class DataStreamer(ABC):
    """Abstract base class to sequentially traverse data based on time.

    Provides an iterable interface for streaming data points at a fixed sample rate.
    Subclasses must implement the `approx_duration` property and the `stream` method.

    Attributes:
        name (str): Name of the DataStreamer, for identification.
        sample_rate (float): The rate in Hz at which to sample and yield data.
    """
    def __init__(self, name: str, sample_rate: float = 30.0) -> None:
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")
        self.name = name
        self.sample_rate = sample_rate
        self._timestep = 1. / sample_rate
        self._clock = 0.0
    
    @property
    @abstractmethod
    def approx_duration(self) -> float:
        """Approximate duration of the datastream in seconds."""
        raise NotImplementedError("Subclasses must implement approx_duration")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata about the data stream. Override in subclasses if needed."""
        return {}

    def __iter__(self) -> 'DataStreamer':
        return self

    @abstractmethod
    def stream(self) -> Any:
        """Return the data point at the current clock time.

        Returns:
            The data point corresponding to the internal clock time.
        Raises:
            StopIteration: when the end of the stream is reached.
        """
        raise NotImplementedError("Subclasses must implement stream()")

    def __next__(self) -> Tuple[float, Any]:
        """Retrieve the next item in the sequence based on the sample_rate setting."""
        time = self._clock
        data = self.stream()
        self._clock += self._timestep
        return time, data


class SizedStreamerProtocol(Protocol):
    @property
    def size(self) -> Tuple[int, int]:
        """Return (width, height)."""
        raise NotImplementedError("Size needs to be implemented for a SizedStreamerProtocol.")


class KnownDurationProtocol(Protocol):
    @property
    @abstractmethod
    def duration(self) -> float:
        """Subclasses must implement."""
        raise NotImplementedError("Duration needs to be implemented for a KnownDurationProtocol.")

    @property
    def approx_duration(self) -> float:
        return self.duration


class StaticMixin(ABC):
    """Mixin for data streamers that always return the same data (static)."""
    def __init__(self, *args, **kwargs):
        self._cached_data = None
        super().__init__(*args, **kwargs)

    @property
    def approx_duration(self) -> float:
        """Static data streams are assumed to be infinite in duration."""
        return float('inf')

    @property
    def duration(self) -> float:
        """Static data streams are assumed to be infinite in duration."""
        return float('inf')

    def stream(self) -> Any:
        if self._cached_data is None:
            self._cached_data = self._generate_data()
        return self._cached_data

    @abstractmethod    
    def _generate_data(self) -> Any:
        raise NotImplementedError("Subclasses must implement _generate_data()")
