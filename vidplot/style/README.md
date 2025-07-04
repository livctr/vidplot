# Vidplot Styling System

The vidplot styling system provides a centralized way to configure visual parameters for all renderers, similar to matplotlib's `rcParams` system.

**Example:**
```python
from vidplot.streamers import VideoStreamer, LabelBarStreamer, StaticDataStreamer
from vidplot.renderers import RGBRenderer, LabelBarRenderer, StringRenderer
from vidplot.core.video_canvas import VideoCanvas
from vidplot.style import use_style, rcParams

use_style('dark')
rcParams().font_family = "DejaVu Sans"

vid_streamer = VideoStreamer("video", "input.mp4")
label_streamer = LabelBarStreamer(
    name="labels",
    data_source="labels.csv",
    time="time",
    data="label",
    duration=vid_streamer.duration,
    num_samples=int(vid_streamer.duration * 30),
)
string_streamer = StaticDataStreamer("string_overlay", "Overlay!")

video_renderer = RGBRenderer("video", vid_streamer)
label_bar_renderer = LabelBarRenderer("label_bar", label_streamer, height=32, font_size=18)
string_renderer = StringRenderer("string_overlay", string_streamer, font_color=(255,0,0), font_scale=1.2)

canvas = VideoCanvas(row_gap=5, col_gap=0)
canvas.attach(vid_streamer, video_renderer, grid_row=2, grid_col=1, height=[vid_streamer.size[1]], width=[vid_streamer.size[0]], z_index=0)
canvas.attach(label_streamer, label_bar_renderer, grid_row=1, grid_col=1, height=[32], width=[vid_streamer.size[0]], z_index=0)
canvas.attach(string_streamer, string_renderer, grid_row=2, grid_col=1, height=[vid_streamer.size[1]], width=[vid_streamer.size[0]], z_index=1)
canvas.write("output.mp4", fps=30.0)
```

## Quick Start

```python
from vidplot import rc, use_style, set_font_size

# Set individual parameters
rc('font_scale', 0.8)
rc('line_thickness', 3)

# Apply predefined styles
use_style('dark')

# Use convenience functions
set_font_size(16)
set_line_width(4)
```

## Main Functions

### `rc(key, value=None)`
Get or set a style parameter.

```python
# Get current value
current_font_scale = rc('font_scale')

# Set new value
rc('font_scale', 0.8)
```

### `rcParams()`
Get the global style configuration object.

```python
config = rcParams()
print(config.font_scale)
print(config.line_thickness)
```

### `use_style(style_name)`
Apply a predefined style configuration.

Available styles:
- `'default'`: Default style
- `'dark'`: Dark theme with light text on dark background
- `'minimal'`: Minimal style with reduced visual elements
- `'high_contrast'`: High contrast black and white style

### `rc_context(rc_dict)`
Context manager for temporarily changing style parameters.

```python
with rc_context({'font_scale': 0.8, 'line_thickness': 3}):
    # Create renderers here - they will use the temporary styling
    renderer = StringRenderer(...)
    # After the context, styling returns to previous state
```

## Convenience Functions

### `set_font_size(size)`
Set the default font size (also updates font_scale).

### `set_line_width(width)`
Set the default line width (also updates line_thickness).

### `set_marker_size(size)`
Set the default marker size (also updates marker_radius).

### `set_color_scheme(colors)`
Set the color scheme for label bars and other multi-color elements.

## Available Parameters

### Font Settings
- `font_family`: Font family (default: "sans-serif")
- `font_size`: Font size in points (default: 12)
- `font_scale`: Font scale factor (default: 0.5)
- `font_color`: Font color as RGB tuple (default: (0, 0, 0))
- `font_thickness`: Font thickness (default: 1)
- `font_face`: OpenCV font face (default: "FONT_HERSHEY_SIMPLEX")

### Line Settings
- `line_width`: Line width (default: 2)
- `line_thickness`: Line thickness (default: 2)
- `line_color`: Line color as RGB tuple (default: (0, 0, 255))

### Marker/Point Settings
- `marker_size`: Marker size (default: 3)
- `marker_radius`: Marker radius (default: 3)
- `marker_color`: Marker color as RGB tuple (default: (255, 0, 0))
- `marker_thickness`: Marker thickness, -1 for filled (default: -1)

### Box Settings
- `box_thickness`: Box line thickness (default: 2)
- `box_color`: Box color as RGB tuple (default: (0, 255, 0))
- `box_alpha`: Box transparency (default: 1.0)

### Text Settings
- `text_color`: Text color as RGB tuple (default: (0, 0, 0))
- `text_scale`: Text scale factor (default: 0.5)
- `text_thickness`: Text thickness (default: 1)
- `text_background_color`: Text background color (default: None)

### Progress Bar Settings
- `progress_bar_color`: Progress bar color as RGB tuple (default: (0, 0, 255))
- `progress_bar_thickness`: Progress bar thickness (default: 2)

### Label Bar Settings
- `label_bar_height`: Label bar height in pixels (default: 20)
- `label_bar_colors`: List of RGB color tuples for label bars

### Background Settings
- `background_color`: Background color as RGB tuple (default: (255, 255, 255))
- `canvas_background`: Canvas background color as RGB tuple (default: (255, 255, 255))

### Grid Settings
- `grid_color`: Grid color as RGB tuple (default: (180, 180, 180))
- `grid_thickness`: Grid line thickness (default: 1)
- `grid_gap`: Grid gap in pixels (default: 0)

### Other Settings
- `confidence_threshold`: Confidence threshold for keypoints (default: 0.0)
- `overlay_alpha`: Alpha blending for overlays (default: 0.7)
- `default_fps`: Default frames per second (default: 30.0)

## Examples

### Basic Usage
```python
from vidplot import rc, use_style

# Apply dark theme
use_style('dark')

# Customize specific parameters
rc('font_scale', 0.8)
rc('line_thickness', 3)
rc('marker_radius', 5)
```

### Context Manager
```python
from vidplot import rc_context

# Temporary styling for specific renderers
with rc_context({
    'font_scale': 0.8,
    'line_thickness': 3,
    'background_color': (240, 240, 240)
}):
    # Create renderers with temporary styling
    renderer = StringRenderer(...)
```

### Custom Color Scheme
```python
from vidplot import set_color_scheme

# Define custom colors
colors = [
    (255, 100, 100),  # Light red
    (100, 255, 100),  # Light green
    (100, 100, 255),  # Light blue
    (255, 255, 100),  # Light yellow
]
set_color_scheme(colors)
```

### Save and Load Configuration
```python
from vidplot import rcParams

# Save current configuration
config = rcParams()
config_dict = config.to_dict()

# Load configuration
new_config = StyleConfig.from_dict(config_dict)
```

## Integration with Renderers

The styling system is designed to work with existing renderers. Renderers can access the global style configuration through `rcParams()` and use the values as defaults for their parameters.

For example, a renderer might use:
```python
from vidplot import rcParams

class MyRenderer:
    def __init__(self, font_scale=None, line_thickness=None):
        config = rcParams()
        self.font_scale = font_scale or config.font_scale
        self.line_thickness = line_thickness or config.line_thickness
```

This allows renderers to have sensible defaults while still allowing users to override specific parameters when needed. 