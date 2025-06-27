from typing import Any, Dict
from . import DataStreamer, StaticMixin

class StaticStreamer(DataStreamer, StaticMixin):
    """
    A static data streamer that returns a fixed value for each call.
    Inherits from StaticMixin for infinite duration behavior.
    """
    def __init__(
        self,
        name: str,
        value: Any,
        sample_rate: float = 30.0,
    ):
        super().__init__(name=name, sample_rate=sample_rate)
        self._value = value

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "value_type": type(self._value).__name__,
            "static": True,
        }

    def _generate_data(self) -> Any:
        """Return the fixed value."""
        return self._value
