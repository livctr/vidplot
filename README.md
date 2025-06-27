# VidPlot

VidPlot is a Python package for visualizing and annotating videos, designed for computer vision and data science workflows. It provides tools to overlay annotations, visualize object tracking, and interactively explore video datasets.

## Features
- Overlay bounding boxes, labels, and custom annotations on videos
- Support for OpenCV and optional Decord backend
- Easy integration with pandas DataFrames and numpy arrays
- Progress bars and batch processing with tqdm
- Interactive visualization with matplotlib

## Installation

```bash
# Basic install
pip install .

# With optional decord support
pip install .[decord]

# Or av support
pip install .[av]
```

## Usage
See the [demo](demo/) folder for example Jupyter notebooks on how to use VidPlot for your projects.

## Development
Run tests with:
```bash
pytest
```

Format code with:
```bash
black .
```

## License
MIT 