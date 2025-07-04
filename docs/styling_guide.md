# Vidplot Styling Guide

VidPlot uses a grid-based layout system via `VideoCanvas`. Attach streamers and renderers to grid cells, and style them globally or per-renderer.

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

---

Disclaimer: Pretty much fully LLM-generated.

Vidplot provides a centralized styling system similar to matplotlib's `rcParams` that allows you to configure visual parameters globally and override them locally for individual renderers.

## Overview

The styling system consists of:
- **Global Configuration**: Centralized defaults via `vidplot.style.rcParams()`
- **Local Overrides**: Individual renderer customization via `**kwargs`
- **Predefined Themes**: Ready-to-use style collections
- **Context Management**: Temporary styling changes

## Global Styling Configuration

### Basic Usage

```python
from vidplot import rc, rcParams, use_style

# Get current global configuration
config = rcParams()
print(f"Current font scale: {config.font_scale}")

# Change global defaults
rc('font_scale', 1.2)
rc('font_color', (255, 0, 0))  # Red text

# Apply predefined themes
use_style('dark')      # Dark theme
use_style('minimal')   # Minimal styling
use_style('default')   # Reset to defaults
```

### Available Global Parameters

| Category | Parameter | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| **Font** | `font_family` | str | "sans-serif" | Font family name |
| | `font_size` | int | 12 | Base font size |
| | `font_scale` | float | 0.5 | Font scaling factor |
| | `font_color` | tuple | (0,0,0) | RGB text color |
| | `font_thickness` | int | 1 | Font thickness |
| | `font_face` | str | "FONT_HERSHEY_SIMPLEX" | OpenCV font face |
| **String** | `string_line_type` | str | "LINE_AA" | OpenCV line type |
| | `string_num_expected_lines` | int | 1 | Lines for size calculation |
| | `string_float_precision` | int | 3 | Decimal places for floats |
| **Line** | `line_width` | int | 2 | Line width |
| | `line_thickness` | int | 2 | Line thickness |
| | `line_color` | tuple | (0,0,255) | RGB line color |
| **Marker** | `marker_size` | int | 3 | Marker size |
| | `marker_radius` | int | 3 | Marker radius |
| | `marker_color` | tuple | (255,0,0) | RGB marker color |
| | `marker_thickness` | int | -1 | Marker thickness (-1=filled) |
| **Box** | `box_thickness` | int | 2 | Box border thickness |
| | `box_color` | tuple | (0,255,0) | RGB box color |
| | `box_alpha` | float | 1.0 | Box transparency |
| **Label Bar** | `label_bar_height` | int | 20 | Height of label bars |
| | `label_bar_colors` | list | [...] | Color scheme for label bars |
| | `progress_bar_color` | tuple | (0,0,255) | RGB progress bar color |
| | `progress_bar_thickness` | int | 2 | Progress bar thickness |
| **Confidence** | `confidence_threshold` | float | 0.0 | Min confidence for keypoints |

## Renderer-Specific Styling with **kwargs

All renderers that support styling accept `**kwargs` to override global defaults. The available parameters depend on the renderer type.

### StringRenderer Parameters

```python
from vidplot.renderers import StringRenderer
from vidplot import StaticDataStreamer

streamer = StaticDataStreamer("text", "Hello World")

# Use all global defaults
renderer = StringRenderer("default", streamer)

# Override specific parameters
renderer = StringRenderer("custom", streamer,
                         font_scale=1.5,           # Larger text
                         font_color=(255, 0, 0),   # Red color
                         thickness=3,              # Bold text
                         float_precision=2)        # Format floats
```

**Available kwargs for StringRenderer:**
- `font_face`: OpenCV font face constant (e.g., `cv2.FONT_HERSHEY_SIMPLEX`)
- `font_scale`: Font size scaling factor (float)
- `font_color`: Text color as RGB tuple `(int, int, int)`
- `thickness`: Font thickness (int)
- `line_type`: OpenCV line type (e.g., `cv2.LINE_AA`)
- `num_expected_lines`: Estimated lines for size calculation (int)
- `float_precision`: Decimal places for float values, `None` to disable (int or None)

### BoxRenderer Parameters

```python
from vidplot.renderers import BoxRenderer
from vidplot import StaticDataStreamer

streamer = StaticDataStreamer("boxes", box_data)
id_to_color = {1: (255, 0, 0), 2: (0, 255, 0)}  # Red, Green

# Use all global defaults
renderer = BoxRenderer("boxes", streamer, id_to_color)

# Override specific parameters
renderer = BoxRenderer("boxes", streamer, id_to_color,
                      line_thickness=3,              # Thicker borders
                      font_scale=0.8)               # Larger labels

# Use different coordinate formats (explicit parameters)
renderer = BoxRenderer("boxes", streamer, id_to_color,
                      label_box=False,              # No labels
                      box_representation_format="xywh",  # Different format
                      resize_mode="fit")
```

**Available kwargs for BoxRenderer:**
- `line_thickness`: Thickness of box borders (int)
- `font_scale`: Font size scaling for labels (float)

**Explicit constructor parameters:**
- `label_box`: Whether to draw labels on boxes (bool)
- `box_representation_format`: 'xyxy' or 'xywh' format (str)
- `resize_mode`: 'stretch', 'fit', or 'center' for coordinate mapping (str)

### LabelBarRenderer Parameters

```python
from vidplot.renderers import LabelBarRenderer
from vidplot import LabelBarStreamer

streamer = LabelBarStreamer("labels", label_data)

# Use all global defaults with auto-generated colors
renderer = LabelBarRenderer("labels", streamer)

# Override specific parameters
renderer = LabelBarRenderer("labels", streamer,
                           height=30,                    # Taller bar
                           progress_bar_color=(255, 0, 0), # Red progress
                           progress_thickness=4)         # Thicker progress

# Custom label colors
label_to_color = {1: (255, 0, 0), 2: (0, 255, 0)}
renderer = LabelBarRenderer("labels", streamer,
                           label_to_color=label_to_color,
                           font_scale=0.8)              # Larger text

# Disable text overlay
renderer = LabelBarRenderer("labels", streamer,
                           write_sampled_data_str=False)
```

**Available kwargs for LabelBarRenderer:**
- `height`: Height of the label bar in pixels (int)
- `progress_bar_color`: RGB color for the progress bar (tuple)
- `progress_thickness`: Thickness of the progress bar (int)
- `font_scale`: Font size scaling for text (float)
- `font_color`: RGB color for text (tuple)
- `thickness`: Font thickness for text (int)

**Explicit constructor parameters:**
- `label_to_color`: Optional mapping from label IDs to RGB colors (dict or None)
- `write_sampled_data_str`: Whether to display text overlay (bool)

### COCOKeypointsRenderer Parameters

```python
from vidplot.renderers import COCOKeypointsRenderer
from vidplot import StaticDataStreamer

streamer = StaticDataStreamer("keypoints", pose_data)

# Use all global defaults
renderer = COCOKeypointsRenderer("keypoints", streamer)

# Override specific parameters
renderer = COCOKeypointsRenderer("keypoints", streamer,
                                color=(255, 0, 0),      # Red keypoints
                                radius=5,               # Larger circles
                                draw_labels=True,       # Show labels
                                confidence_threshold=0.5)  # Higher threshold

# Custom keypoint labels
labels = {0: 'nose', 1: 'left_eye', 2: 'right_eye'}
renderer = COCOKeypointsRenderer("keypoints", streamer,
                                keypoint_labels=labels,
                                font_scale=0.6)
```

**Available kwargs for COCOKeypointsRenderer:**
- `color`: RGB color for keypoint circles (tuple)
- `radius`: Radius of keypoint circles (int)
- `thickness`: Circle thickness, -1 for filled (int)
- `draw_labels`: Whether to draw labels on keypoints (bool)
- `keypoint_labels`: Mapping from index to string label (dict)
- `font_scale`: Font size scaling for labels (float)
- `font_color`: RGB color for label text (tuple)
- `font_thickness`: Font thickness for labels (int)
- `font_face`: OpenCV font face constant (int)
- `confidence_threshold`: Minimum confidence to show keypoints (float)
- `assume_normalized`: Force normalized/pixel input detection (bool or None)

### COCOKeypoints3DRenderer Parameters

```python
from vidplot.renderers import COCOKeypoints3DRenderer
from vidplot import StaticDataStreamer

streamer = StaticDataStreamer("3d_keypoints", pose_3d_data)

# Use all global defaults
renderer = COCOKeypoints3DRenderer("3d_keypoints", streamer)

# Override specific parameters
renderer = COCOKeypoints3DRenderer("3d_keypoints", streamer,
                                   marker_color='blue',      # Blue markers
                                   marker_size=30)          # Larger markers

# Custom 3D view (explicit parameters)
renderer = COCOKeypoints3DRenderer("3d_keypoints", streamer,
                                   figsize=(6, 6),           # Larger plot
                                   elev=45,                  # 45° elevation
                                   azim=-45,                # -45° azimuth
                                   confidence_threshold=0.5) # Higher threshold
```

**Available kwargs for COCOKeypoints3DRenderer:**
- `marker_color`: Color for 3D keypoint markers (str)
- `marker_size`: Size of 3D keypoint markers (int)

**Explicit constructor parameters:**
- `figsize`: Figure size for matplotlib plot (tuple)
- `elev`: Elevation angle for 3D view (int)
- `azim`: Azimuth angle for 3D view (int)
- `confidence_threshold`: Minimum confidence to show keypoints (float)

### Other Renderer Parameters

Different renderers support different styling parameters. Check each renderer's documentation for available kwargs.

## Context Management

Use `rc_context()` for temporary styling changes that automatically revert:

```python
from vidplot import rc_context

# Temporary styling changes
with rc_context({
    'font_scale': 0.8,
    'font_color': (0, 0, 255),  # Blue
    'font_thickness': 2,
}):
    renderer = StringRenderer("temp", streamer)
    # This renderer uses the temporary styling
    renderer.render(data, bbox, canvas)

# Outside the context, global defaults are restored
renderer2 = StringRenderer("normal", streamer)
# This renderer uses the original global defaults
```

## Predefined Themes

Vidplot comes with several predefined themes:

### Dark Theme
```python
use_style('dark')
# Features: Dark background, light text, muted colors
```

### Minimal Theme
```python
use_style('minimal')
# Features: Smaller fonts, thinner lines, reduced visual elements
```

### High Contrast Theme
```python
use_style('high_contrast')
# Features: Black text on white background, bold elements
```

### Default Theme
```python
use_style('default')
# Features: Standard styling with good readability
```

## Best Practices

### 1. Set Global Defaults First
```python
# Set up your preferred global styling
use_style('dark')
rc('font_scale', 0.8)

# Create renderers (they'll use your defaults)
renderer1 = StringRenderer("title", title_streamer)
renderer2 = BoxRenderer("boxes", box_streamer, id_to_color)
renderer3 = LabelBarRenderer("labels", label_streamer)
```

### 2. Use kwargs for Exceptions
```python
# Most elements use global defaults
renderer1 = StringRenderer("normal", streamer)
renderer2 = BoxRenderer("normal", box_streamer, id_to_color)
renderer3 = LabelBarRenderer("normal", label_streamer)

# Special cases override specific parameters
renderer4 = StringRenderer("highlight", streamer,
                          font_color=(255, 255, 0),  # Yellow
                          thickness=3)               # Bold
```

### 3. Use Context Managers for Temporary Changes
```python
# Temporary styling for a specific section
with rc_context({'font_scale': 1.2}):
    renderer = StringRenderer("large", streamer)
    # Process with larger text
```

### 4. Format Dynamic Data
```python
# Format numbers with specific precision
number_streamer = StaticDataStreamer("value", 42.123456)
renderer = StringRenderer("formatted", number_streamer,
                         float_precision=2)  # Shows "42.12"
```

### 5. Coordinate Multiple Renderers
```python
# Consistent styling across renderers
use_style('dark')
rc('confidence_threshold', 0.5)

# All renderers use the same theme
keypoints = COCOKeypointsRenderer("pose", pose_streamer)
boxes = BoxRenderer("detections", box_streamer, id_to_color)
labels = LabelBarRenderer("labels", label_streamer,
                         height=25)  # Taller than global
text = StringRenderer("info", info_streamer)
```

## Advanced Usage

### Custom Style Configuration
```python
from vidplot.style import StyleConfig

# Create a custom configuration
custom_style = StyleConfig()
custom_style.font_scale = 1.0
custom_style.font_color = (128, 128, 128)  # Gray
custom_style.line_thickness = 1

# Apply it temporarily
with rc_context(custom_style.to_dict()):
    renderer = StringRenderer("custom", streamer)
```

### Style Inheritance
```python
# Base styling
use_style('dark')

# Override for specific renderer
renderer = StringRenderer("special", streamer,
                         font_scale=1.5,      # Override global
                         font_color=(255, 0, 0))  # Override global

# Other renderers still use dark theme defaults
renderer2 = StringRenderer("normal", streamer)
```

### Multi-Renderer Coordination
```python
# Set up consistent styling for a complex visualization
use_style('minimal')
rc('confidence_threshold', 0.7)

# Create coordinated renderers
video = RGBRenderer("video", video_streamer)
keypoints = COCOKeypointsRenderer("pose", pose_streamer,
                                 draw_labels=True,
                                 confidence_threshold=0.8)  # Higher than global
boxes = BoxRenderer("detections", box_streamer, id_to_color,
                   line_thickness=1)  # Thinner than global
labels = LabelBarRenderer("labels", label_streamer,
                         height=25)  # Taller than global
text = StringRenderer("info", info_streamer,
                     font_scale=0.6)  # Smaller than global
```

## Troubleshooting

### Common Issues

1. **Colors not appearing correctly**
   - Ensure colors are RGB tuples: `(255, 0, 0)` not `(255, 0, 0, 255)`
   - Check that values are integers in range 0-255

2. **Font too small/large**
   - Adjust `font_scale` parameter
   - Use `set_font_size()` convenience function

3. **Text not visible**
   - Check `font_color` against background
   - Verify `thickness` is not 0

4. **Float formatting not working**
   - Set `float_precision` to an integer (not None)
   - Ensure data is actually a float

5. **Keypoints not showing**
   - Check `confidence_threshold` value
   - Verify `radius` and `thickness` are appropriate

6. **3D plots not rendering**
   - Ensure matplotlib is installed
   - Check `figsize` is reasonable
   - Verify `elev` and `azim` angles are valid

7. **Label bar colors not consistent**
   - Check `label_to_color` mapping
   - Verify global `label_bar_colors` configuration
   - Ensure label IDs are integers

### Debugging Style Parameters
```python
# Check current global configuration
config = rcParams()
print(f"Font scale: {config.font_scale}")
print(f"Font color: {config.font_color}")
print(f"Confidence threshold: {config.confidence_threshold}")
print(f"Label bar height: {config.label_bar_height}")

# Check specific parameter
current_scale = rc('font_scale')
print(f"Current font scale: {current_scale}")
```

This styling system provides flexible, consistent visual configuration across all vidplot renderers while maintaining the ability to customize individual elements as needed. 