"""
Vidplot styling configuration system.

This module provides a centralized way to configure visual styling parameters
for vidplot renderers, similar to matplotlib's rcParams system.
"""

from typing import Dict, Any, Union
import copy


class StyleConfig:
    """
    Centralized styling configuration for vidplot renderers.

    This class manages default values for visual parameters like fonts, colors,
    line widths, marker sizes, etc. It provides a similar interface to matplotlib's
    rcParams system.
    """

    def __init__(self):
        # Font settings
        self.font_family = "sans-serif"
        self.font_size = 12
        self.font_scale = 0.5
        self.font_color = (0, 0, 0)  # Black in RGB
        self.font_thickness = 1
        self.font_face = "FONT_HERSHEY_SIMPLEX"  # OpenCV font face

        # String rendering settings
        self.string_line_type = "LINE_AA"  # OpenCV line type
        self.string_num_expected_lines = 1
        self.string_float_precision = 3

        # Line settings
        self.line_width = 2
        self.line_thickness = 2
        self.line_color = (0, 0, 255)  # Blue in RGB

        # Marker/point settings
        self.marker_size = 3
        self.marker_radius = 3
        self.marker_color = (255, 0, 0)  # Red in RGB
        self.marker_thickness = -1  # -1 for filled

        # Box settings
        self.box_thickness = 2
        self.box_color = (0, 255, 0)  # Green in RGB
        self.box_alpha = 1.0

        # Text settings
        self.text_color = (0, 0, 0)  # Black in RGB
        self.text_scale = 0.5
        self.text_thickness = 1
        self.text_background_color = None  # None for transparent

        # Progress bar settings
        self.progress_bar_color = (0, 0, 255)  # Blue in RGB
        self.progress_bar_thickness = 2

        # Label bar settings
        self.label_bar_height = 20
        self.label_bar_colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),  # Dark Red
            (0, 128, 0),  # Dark Green
            (0, 0, 128),  # Dark Blue
            (128, 128, 0),  # Olive
        ]

        # Background settings
        self.background_color = (255, 255, 255)  # White in RGB
        self.canvas_background = (255, 255, 255)  # White in RGB

        # Grid settings
        self.grid_color = (180, 180, 180)  # Light gray in RGB
        self.grid_thickness = 1
        self.grid_gap = 0

        # Confidence threshold for keypoints
        self.confidence_threshold = 0.0

        # Alpha blending for overlays
        self.overlay_alpha = 0.7

        # Animation settings
        self.default_fps = 30.0

    def update(self, **kwargs):
        """
        Update configuration parameters.

        Parameters
        ----------
        **kwargs : dict
            Parameters to update. Valid keys include all attributes of this class.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown style parameter: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration parameter.

        Parameters
        ----------
        key : str
            Parameter name
        default : any, optional
            Default value if parameter doesn't exist

        Returns
        -------
        any
            Parameter value
        """
        return getattr(self, key, default)

    def copy(self) -> "StyleConfig":
        """
        Create a copy of the current configuration.

        Returns
        -------
        StyleConfig
            A new StyleConfig instance with the same parameters
        """
        return copy.deepcopy(self)

    def reset(self):
        """Reset all parameters to their default values."""
        self.__init__()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration
        """
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StyleConfig":
        """
        Create a StyleConfig from a dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary with configuration parameters

        Returns
        -------
        StyleConfig
            New StyleConfig instance
        """
        config = cls()
        config.update(**config_dict)
        return config


# Global style configuration instance
_style_config = StyleConfig()


def rcParams() -> StyleConfig:
    """
    Get the global style configuration.

    Returns
    -------
    StyleConfig
        The global style configuration instance
    """
    return _style_config


def rc(key: str, value: Any = None):
    """
    Get or set a style parameter.

    Parameters
    ----------
    key : str
        Parameter name
    value : any, optional
        New value for the parameter. If None, returns the current value.

    Returns
    -------
    any
        Current value of the parameter (if value is None)
    """
    if value is None:
        return _style_config.get(key)
    else:
        _style_config.update(**{key: value})
        return value


def rc_context(rc_dict: Dict[str, Any]):
    """
    Context manager for temporarily changing style parameters.

    Parameters
    ----------
    rc_dict : dict
        Dictionary of parameters to temporarily set

    Yields
    ------
    StyleConfig
        The modified style configuration
    """
    original_config = _style_config.copy()
    try:
        _style_config.update(**rc_dict)
        yield _style_config
    finally:
        _style_config.__dict__.update(original_config.__dict__)


def use_style(style_name: str):
    """
    Apply a predefined style configuration.

    Parameters
    ----------
    style_name : str
        Name of the style to apply. Currently supported:
        - 'default': Default style
        - 'dark': Dark theme
        - 'minimal': Minimal style with reduced visual elements
        - 'high_contrast': High contrast style
    """
    if style_name == "default":
        _style_config.reset()
    elif style_name == "dark":
        _style_config.update(
            background_color=(30, 30, 30),
            canvas_background=(30, 30, 30),
            font_color=(255, 255, 255),
            text_color=(255, 255, 255),
            grid_color=(80, 80, 80),
            line_color=(100, 150, 255),
            marker_color=(255, 100, 100),
            box_color=(100, 255, 100),
        )
    elif style_name == "minimal":
        _style_config.update(
            font_scale=0.4,
            font_thickness=1,
            line_thickness=1,
            box_thickness=1,
            marker_radius=2,
            grid_gap=5,
            overlay_alpha=0.5,
        )
    elif style_name == "high_contrast":
        _style_config.update(
            font_color=(0, 0, 0),
            text_color=(0, 0, 0),
            background_color=(255, 255, 255),
            canvas_background=(255, 255, 255),
            line_color=(0, 0, 0),
            marker_color=(255, 0, 0),
            box_color=(0, 0, 0),
            grid_color=(0, 0, 0),
            font_thickness=2,
            line_thickness=3,
            box_thickness=3,
        )
    else:
        raise ValueError(f"Unknown style: {style_name}")


# Convenience functions for common operations
def set_font_size(size: Union[int, float]):
    """Set the default font size."""
    rc("font_size", size)
    rc("font_scale", size / 24.0)  # Approximate scaling


def set_line_width(width: Union[int, float]):
    """Set the default line width."""
    rc("line_width", width)
    rc("line_thickness", int(width))


def set_marker_size(size: Union[int, float]):
    """Set the default marker size."""
    rc("marker_size", size)
    rc("marker_radius", int(size))


def set_color_scheme(colors: list):
    """Set the color scheme for label bars and other multi-color elements."""
    rc("label_bar_colors", colors)


# Export the main interface
__all__ = [
    "StyleConfig",
    "rcParams",
    "rc",
    "rc_context",
    "use_style",
    "set_font_size",
    "set_line_width",
    "set_marker_size",
    "set_color_scheme",
]
