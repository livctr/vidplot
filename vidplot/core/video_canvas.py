import os
from typing import List, Tuple, Dict, Optional, Any, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .renderer import Renderer
from .streamer import DataStreamer, StaticDataStreamer


class VideoCanvas:
    def __init__(
        self,
        row_gap: int = 0,
        col_gap: int = 0,
        rounding_decimals_for_time_sync: int = 3,
        stream_method: str = "nearest_neighbor",
    ):
        """
        VideoCanvas manages the layout and rendering of multiple streamers and renderers on a grid.
        The grid is dynamically determined by the attached renderers and their specified
        rows/columns.

        Args:
            row_gap: Pixel gap between grid rows
            col_gap: Pixel gap between grid columns
            rounding_decimals_for_time_sync: Number of decimals for time sync
            stream_method: Streaming method for dynamic data
        """
        self.row_gap = row_gap
        self.col_gap = col_gap
        self.stream_method = stream_method
        self.round_decimals = rounding_decimals_for_time_sync
        self._attachments: List[Dict[str, Any]] = (
            []
        )  # Each dict: streamer, renderer, grid_row, grid_col, height, width, z_index
        self.streamers: Dict[str, DataStreamer] = {}
        self.renderers: Dict[str, Renderer] = {}
        self.routes: List[Tuple[str, str]] = []
        self._cell_coords = None
        self._canvas_shape = None
        self._row_heights = []
        self._col_widths = []
        self._renderer_info: Dict[str, Dict[str, Any]] = {}  # Fast lookup for renderer info

    def attach(
        self,
        streamer: DataStreamer,
        renderer: Renderer,
        grid_row: Union[int, Tuple[int, int], None] = None,
        grid_col: Union[int, Tuple[int, int], None] = None,
        height: Optional[List[int]] = None,
        width: Optional[List[int]] = None,
        z_index: int = 0,
    ):
        """
        Attach a streamer and renderer to the canvas at a specified grid location.
        grid_row/grid_col can be int (interpreted as (i,i)) or tuple (start, end).
        height/width are lists of ints for each row/col spanned, or None to infer
        from renderer.default_size.
        z_index controls draw order (higher is on top).
        Prevents duplicate streamer or renderer names.
        """
        # Normalize grid_row/grid_col
        if grid_row is not None:
            if isinstance(grid_row, int):
                grid_row = (grid_row, grid_row)
            elif not (isinstance(grid_row, tuple) and len(grid_row) == 2):
                raise ValueError("grid_row must be int or tuple of (start, end)")
        if grid_col is not None:
            if isinstance(grid_col, int):
                grid_col = (grid_col, grid_col)
            elif not (isinstance(grid_col, tuple) and len(grid_col) == 2):
                raise ValueError("grid_col must be int or tuple of (start, end)")
        # Infer height/width if not provided
        n_rows = (grid_row[1] - grid_row[0] + 1) if grid_row else 1
        n_cols = (grid_col[1] - grid_col[0] + 1) if grid_col else 1
        if height is not None and len(height) != n_rows:
            raise ValueError(f"height must have length {n_rows} for grid_row {grid_row}")
        if width is not None and len(width) != n_cols:
            raise ValueError(f"width must have length {n_cols} for grid_col {grid_col}")
        # If not provided, try to infer from renderer.default_size
        if height is None or width is None:
            dsize = renderer.default_size if hasattr(renderer, "default_size") else (None, None)
            if height is None:
                if dsize[1] is not None:
                    height = [dsize[1]] * n_rows
                else:
                    raise ValueError(
                        "height must be provided or renderer.default_size[1] must be set"
                    )
            if width is None:
                if dsize[0] is not None:
                    width = [dsize[0]] * n_cols
                else:
                    raise ValueError(
                        "width must be provided or renderer.default_size[0] must be set"
                    )
        # Register streamer and renderer
        self.streamers[streamer.name] = streamer
        self.renderers[renderer.name] = renderer
        attachment = {
            "streamer": streamer,
            "renderer": renderer,
            "grid_row": grid_row,
            "grid_col": grid_col,
            "height": height,
            "width": width,
            "z_index": z_index,
        }
        self._attachments.append(attachment)
        self._renderer_info[renderer.name] = attachment  # Fast lookup by renderer name
        self._recompute_layout()
        # Update routes (sorted by z_index)
        self.routes = [
            (a["streamer"].name, a["renderer"].name)
            for a in sorted(self._attachments, key=lambda a: a["z_index"])
        ]

    def clear(self):
        """
        Erase all current streamers, renderers, and attachments from the canvas.
        """
        self._attachments.clear()
        self.streamers.clear()
        self.renderers.clear()
        self.routes.clear()
        self._cell_coords = None
        self._canvas_shape = None
        self._row_heights = []
        self._col_widths = []
        self._renderer_info.clear()  # Also clear the fast lookup dict

    def _recompute_layout(self):
        """
        Compute the grid layout: row heights, col widths, cell coordinates, and canvas shape.
        Each cell's height is the max over all renderers spanning that row, and likewise for width.
        Uses row_gap and col_gap for spacing.
        """
        # Find max row/col indices
        all_rows = []
        all_cols = []
        for a in self._attachments:
            if a["grid_row"]:
                all_rows.extend(range(a["grid_row"][0], a["grid_row"][1] + 1))
            if a["grid_col"]:
                all_cols.extend(range(a["grid_col"][0], a["grid_col"][1] + 1))
        n_rows = max(all_rows) if all_rows else 1
        n_cols = max(all_cols) if all_cols else 1
        # For each row/col, find max height/width
        row_heights = [0] * n_rows
        col_widths = [0] * n_cols
        for a in self._attachments:
            if a["grid_row"] and a["height"]:
                for i, r in enumerate(range(a["grid_row"][0], a["grid_row"][1] + 1)):
                    row_heights[r - 1] = max(row_heights[r - 1], a["height"][i])
            if a["grid_col"] and a["width"]:
                for i, c in enumerate(range(a["grid_col"][0], a["grid_col"][1] + 1)):
                    col_widths[c - 1] = max(col_widths[c - 1], a["width"][i])
        self._row_heights = row_heights
        self._col_widths = col_widths
        # Compute cell coordinates
        coords = {}
        y = 0
        for i, row_h in enumerate(row_heights):
            x = 0
            for j, col_w in enumerate(col_widths):
                coords[(i + 1, j + 1)] = (x, y, x + col_w, y + row_h)
                x += col_w + self.col_gap
            y += row_h + self.row_gap
        self._cell_coords = coords
        # Compute canvas shape
        height = sum(row_heights) + (len(row_heights) - 1) * self.row_gap
        width = sum(col_widths) + (len(col_widths) - 1) * self.col_gap
        self._canvas_shape = (height, width, 3)

    def show_layout(self, outpath: str):
        canvas = np.ones(self._canvas_shape, dtype=np.uint8) * 255
        # Draw grid cells as dashed lines
        for i, row_h in enumerate(self._row_heights):
            for j, col_w in enumerate(self._col_widths):
                x1, y1, x2, y2 = self._cell_coords[(i + 1, j + 1)]
                # Dashed rectangle
                for k in range(x1, x2, 10):
                    cv2.line(
                        canvas,
                        (k, y1),
                        (min(k + 5, x2), y1),
                        (180, 180, 180),
                        1,
                    )
                    cv2.line(
                        canvas,
                        (k, y2 - 1),
                        (min(k + 5, x2), y2 - 1),
                        (180, 180, 180),
                        1,
                    )
                for k in range(y1, y2, 10):
                    cv2.line(
                        canvas,
                        (x1, k),
                        (x1, min(k + 5, y2)),
                        (180, 180, 180),
                        1,
                    )
                    cv2.line(
                        canvas,
                        (x2 - 1, k),
                        (x2 - 1, min(k + 5, y2)),
                        (180, 180, 180),
                        1,
                    )
                # Label cell
                cv2.putText(
                    canvas,
                    f"({i+1},{j+1})",
                    (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (120, 120, 120),
                    1,
                )
        # Draw renderer bounding boxes (slightly smaller than cells)
        color_map = plt.get_cmap("tab10", len(self.renderers))
        # Group renderers by cell to handle overlaps
        cell_renderers = {}
        for a in self._attachments:
            r = a["renderer"]
            cell = (a["grid_row"], a["grid_col"])
            if cell not in cell_renderers:
                cell_renderers[cell] = []
            cell_renderers[cell].append(r)
        # Draw each group
        for cell, renderers in cell_renderers.items():
            x1, y1 = self._cell_coords[(cell[0][0], cell[1][0])][:2]
            x2, y2 = self._cell_coords[(cell[0][1], cell[1][1])][2:]
            # Make boxes smaller than cell
            margin = 2
            x1, y1, x2, y2 = x1 + margin, y1 + margin, x2 - margin, y2 - margin
            # For multiple renderers in same cell, make them progressively smaller
            for idx, r in enumerate(renderers):
                shrink = idx * 3
                rx1, ry1, rx2, ry2 = (
                    x1 + shrink,
                    y1 + shrink,
                    x2 - shrink,
                    y2 - shrink,
                )
                color = tuple(
                    int(255 * c) for c in color_map(list(self.renderers.values()).index(r))[:3]
                )
                cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), color, 2)
                cv2.putText(
                    canvas,
                    r.name,
                    (rx1 + 5, ry1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
        cv2.imwrite(outpath, canvas)

    def write(self, outpath: str, fourcc_str: str = "mp4v", fps: float = 30.0):
        """
        Write the annotated output to a file. Supports both video and image output.

        Parameters
        ----------
        outpath : str
            Output file path. If it ends with a video extension (e.g., .mp4, .avi), writes a video.
            If it ends with an image extension (e.g., .png, .jpg, .jpeg), writes just the first
            frame as an image.
        fourcc_str : str, optional
            FourCC code for video encoding (default: 'mp4v').
        fps : float, optional
            Frames per second for video output (default: 30.0).
        """

        # Determine output type by file extension
        video_exts = {".mp4", ".avi", ".mov", ".mkv"}
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        ext = os.path.splitext(outpath)[1].lower()
        if ext == "":
            raise ValueError("Got empty extension.")
        is_image = ext in image_exts
        is_video = ext in video_exts
        assert is_image or is_video, f"Got extension {ext}"
        assert not (is_image and is_video), "Cannot write both image and video at the same time."

        height, width, _ = self._canvas_shape
        if is_video:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(outpath, fourcc, fps, (width, height))

        # Separate static and dynamic streamers
        static_streamers = {}
        dynamic_streamers = {}
        static_data = {}

        for name, streamer in self.streamers.items():
            if isinstance(streamer, StaticDataStreamer):
                static_streamers[name] = streamer
                # Cache static data once
                static_data[name] = streamer.stream()
            else:
                dynamic_streamers[name] = streamer

        # Prepare dynamic streamer iterators and buffers
        streamer_iters = {name: iter(s) for name, s in dynamic_streamers.items()}
        streamer_buffers = {}
        streamer_hit_last = {name: False for name in dynamic_streamers}
        streamer_done = {name: False for name in dynamic_streamers}
        no_dynamic_streamers = len(dynamic_streamers) == 0

        # For tqdm bar, use min(duration) if any finite, else 1.0
        # Only consider dynamic streamers for duration calculation
        approx_durations = [s.duration for s in dynamic_streamers.values()]
        # import pdb ; pdb.set_trace()
        bar_duration = min(approx_durations) if approx_durations else 1
        assert bar_duration < float(
            "inf"
        ), "At least one dynamic streamer must have a finite duration."

        n_frames = int(bar_duration * fps)

        orchestrator_time = 0.0
        frame_idx = 0

        # Gather closest data from all streamers
        data_dict = {}
        # Add static data (no iteration needed)
        data_dict.update(static_data)

        with tqdm(total=n_frames, desc="Rendering video") as pbar:
            while True:

                rounded_orchestrator_time = round(orchestrator_time, self.round_decimals)

                # Process dynamic streamers
                for name, it in streamer_iters.items():

                    # Buffer: (prev_time, prev_data), (next_time, next_data)
                    buf = streamer_buffers.get(name, [])

                    # Fill a buffer of length 2 until we reach or pass the orchestrator time
                    while not streamer_hit_last[name] and (
                        not buf or buf[-1][0] < rounded_orchestrator_time
                    ):
                        try:
                            t, d = next(it)
                            if len(buf) == 2:
                                buf.pop(0)
                            buf.append((round(t, self.round_decimals), d))
                        except StopIteration:
                            streamer_hit_last[name] = True
                            break

                    # If the orchestrator time has not passed the last buffered time,
                    # return the closest frame based on the streaming method
                    if buf and buf[-1][0] >= rounded_orchestrator_time:
                        # We have at least one frame past or at orchestrator time
                        if self.stream_method == "locf" or len(buf) == 1:
                            data_dict[name] = buf[0][1]
                        else:
                            t1, d1 = buf[0]
                            t2, d2 = buf[1]
                            if abs(t1 - rounded_orchestrator_time) <= abs(
                                t2 - rounded_orchestrator_time
                            ):
                                data_dict[name] = d1
                            else:
                                data_dict[name] = d2
                    # Otherwise, (1) we've hit the end of the stream or
                    # (2) the buffer doesn't have any data that is >= orchestrator time.
                    # In both cases, we are done with this streamer.
                    else:
                        streamer_done[name] = True

                    # Update the bufer
                    streamer_buffers[name] = buf

                # If any dynamic streamer is done, break
                if any(streamer_done.values()):
                    break

                # Renderers draw in z-order
                canvas = np.ones(self._canvas_shape, dtype=np.uint8) * 255
                for sname, rname in self.routes:
                    r = self.renderers[rname]
                    # Fast lookup for this renderer's attachment info
                    info = self._renderer_info[rname]
                    grid_row = info["grid_row"]
                    grid_col = info["grid_col"]
                    # Get the full bbox spanning from start to end cell
                    x1, y1 = self._cell_coords[(grid_row[0], grid_col[0])][:2]
                    x2, y2 = self._cell_coords[(grid_row[1], grid_col[1])][2:]
                    bbox = (x1, y1, x2, y2)
                    canvas = r.render(data_dict[sname], bbox, canvas)

                canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                if no_dynamic_streamers or is_image:
                    cv2.imwrite(outpath, canvas)
                    return
                else:
                    writer.write(canvas)
                    orchestrator_time += 1.0 / fps
                    frame_idx += 1
                    pbar.update(1)

        if is_video:
            writer.release()
