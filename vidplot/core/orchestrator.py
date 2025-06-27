from typing import List, Tuple, Dict, Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .renderer import Renderer
from .streamer import DataStreamer, StaticMixin


class AnnotationOrchestrator:
    def __init__(
        self,
        grid_template_rows: List[int],
        grid_template_columns: List[int],
        gap: int = 0,
    ):
        self.grid_template_rows = grid_template_rows
        self.grid_template_columns = grid_template_columns
        self.gap = gap
        self.streamers: Dict[str, DataStreamer] = {}
        self.renderers: Dict[str, Renderer] = {}
        self.routes: List[Tuple[str, str]] = []
        self._cell_coords = self._compute_cell_coords()
        self._canvas_shape = self._compute_canvas_shape()

    def _compute_cell_coords(self):
        # Returns a dict: (row, col) -> (x1, y1, x2, y2)
        coords = {}
        y = 0
        for i, row_h in enumerate(self.grid_template_rows):
            x = 0
            for j, col_w in enumerate(self.grid_template_columns):
                coords[(i+1, j+1)] = (x, y, x+col_w, y+row_h)
                x += col_w + self.gap
            y += row_h + self.gap
        return coords

    def _compute_canvas_shape(self):
        height = sum(self.grid_template_rows) + (len(self.grid_template_rows)-1)*self.gap
        width = sum(self.grid_template_columns) + (len(self.grid_template_columns)-1)*self.gap
        return (height, width, 3)

    def set_annotators(
        self,
        streamers: List[DataStreamer],
        renderers: List[Renderer],
        routes: List[Tuple[str, str]]
    ):
        # Register streamers and renderers by name
        self.streamers = {s.name: s for s in streamers}
        self.renderers = {r.name: r for r in renderers}
        # Sort routes by renderer z_index
        def z_index_of_renderer(rname):
            return self.renderers[rname].z_index if rname in self.renderers else 0
        routes_sorted = sorted(routes, key=lambda pair: z_index_of_renderer(pair[1]))
        self.routes = routes_sorted
        # Check that each renderer fits in the grid
        for r in renderers:
            row_start, row_end = r.grid_row
            col_start, col_end = r.grid_column
            if row_start < 1 or row_end > len(self.grid_template_rows):
                raise ValueError(f"Renderer {r.name} row out of bounds.")
            if col_start < 1 or col_end > len(self.grid_template_columns):
                raise ValueError(f"Renderer {r.name} col out of bounds.")

    def show_layout(self, outpath: str):
        canvas = np.ones(self._canvas_shape, dtype=np.uint8) * 255
        # Draw grid cells as dashed lines
        for i, row_h in enumerate(self.grid_template_rows):
            for j, col_w in enumerate(self.grid_template_columns):
                x1, y1, x2, y2 = self._cell_coords[(i+1, j+1)]
                # Dashed rectangle
                for k in range(x1, x2, 10):
                    cv2.line(canvas, (k, y1), (min(k+5, x2), y1), (180,180,180), 1)
                    cv2.line(canvas, (k, y2-1), (min(k+5, x2), y2-1), (180,180,180), 1)
                for k in range(y1, y2, 10):
                    cv2.line(canvas, (x1, k), (x1, min(k+5, y2)), (180,180,180), 1)
                    cv2.line(canvas, (x2-1, k), (x2-1, min(k+5, y2)), (180,180,180), 1)
                # Label cell
                cv2.putText(canvas, f"({i+1},{j+1})", (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)
        # Draw renderer bounding boxes (slightly smaller than cells)
        color_map = plt.get_cmap('tab10', len(self.renderers))
        # Group renderers by cell to handle overlaps
        cell_renderers = {}
        for r in self.renderers.values():
            cell = (r.grid_row, r.grid_column)
            if cell not in cell_renderers:
                cell_renderers[cell] = []
            cell_renderers[cell].append(r)
        # Draw each group
        for cell, renderers in cell_renderers.items():
            x1, y1 = self._cell_coords[(cell[0][0], cell[1][0])][:2]
            x2, y2 = self._cell_coords[(cell[0][1], cell[1][1])][2:]
            # Make boxes smaller than cell
            margin = 5
            x1, y1, x2, y2 = x1+margin, y1+margin, x2-margin, y2-margin
            # For multiple renderers in same cell, make them progressively smaller
            for idx, r in enumerate(renderers):
                shrink = idx * 3
                rx1, ry1, rx2, ry2 = x1+shrink, y1+shrink, x2-shrink, y2-shrink
                color = tuple(int(255*c) for c in color_map(list(self.renderers.values()).index(r))[:3])
                cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), color, 2)
                cv2.putText(canvas, r.name, (rx1+5, ry1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(outpath, canvas)

    def write(self, outpath: str, fourcc_str: str = 'mp4v', fps: float = 30.0, duration: Optional[float] = None):
        height, width, _ = self._canvas_shape
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(outpath, fourcc, fps, (width, height))
        
        # Separate static and dynamic streamers
        static_streamers = {}
        dynamic_streamers = {}
        static_data = {}
        
        for name, streamer in self.streamers.items():
            if isinstance(streamer, StaticMixin):
                static_streamers[name] = streamer
                # Cache static data once
                static_data[name] = streamer.stream()
            else:
                dynamic_streamers[name] = streamer
        
        # Prepare dynamic streamer iterators and buffers
        streamer_iters = {name: iter(s) for name, s in dynamic_streamers.items()}
        streamer_buffers = {}
        streamer_done = {name: False for name in dynamic_streamers}
        
        # For progress bar, use min(approx_duration) if any finite, else 1.0
        # Only consider dynamic streamers for duration calculation
        approx_durations = [s.approx_duration for s in dynamic_streamers.values()]
        finite_durations = [d for d in approx_durations if d < float('inf')]
        bar_duration = min(finite_durations) if finite_durations else 1.0
        n_frames = int(bar_duration * fps)
        
        orchestrator_time = 0.0
        frame_idx = 0

        # Gather closest data from all streamers
        data_dict = {}
        # Add static data (no iteration needed)
        data_dict.update(static_data)
        
        with tqdm(total=n_frames, desc="Rendering video") as pbar:
            while True:

                # Process dynamic streamers
                for name, it in streamer_iters.items():
                    if streamer_done[name]:
                        data_dict[name] = None
                        continue
                    
                    # Buffer: (prev_time, prev_data), (next_time, next_data)
                    buf = streamer_buffers.get(name, None)
                    
                    # Fill buffer if needed
                    while True:
                        if buf is None:
                            try:
                                t1, d1 = next(it)
                                t2, d2 = next(it)
                                buf = [(t1, d1), (t2, d2)]
                            except StopIteration:
                                streamer_done[name] = True
                                data_dict[name] = None
                                break
                        
                        t1, d1 = buf[0]
                        t2, d2 = buf[1]
                        
                        if t2 >= orchestrator_time:
                            # Pick closer
                            if abs(t1 - orchestrator_time) <= abs(t2 - orchestrator_time):
                                data_dict[name] = d1
                            else:
                                data_dict[name] = d2
                            streamer_buffers[name] = buf
                            break
                        else:
                            # Advance buffer
                            try:
                                t_next, d_next = next(it)
                                buf = [buf[1], (t_next, d_next)]
                            except StopIteration:
                                streamer_done[name] = True
                                data_dict[name] = None
                                break
                    
                    streamer_buffers[name] = buf
                
                # If any dynamic streamer is done, break
                if any(streamer_done.values()):
                    break
                
                # Renderers draw in z-order
                canvas = np.ones(self._canvas_shape, dtype=np.uint8) * 255
                for sname, rname in self.routes:
                    r = self.renderers[rname]
                    # Get the full bbox spanning from start to end cell
                    x1, y1 = self._cell_coords[(r.grid_row[0], r.grid_column[0])][:2]
                    x2, y2 = self._cell_coords[(r.grid_row[1], r.grid_column[1])][2:]
                    bbox = (x1, y1, x2, y2)
                    canvas = r.render(data_dict[sname], bbox, canvas)
                
                writer.write(canvas)
                orchestrator_time += 1.0 / fps
                frame_idx += 1
                pbar.update(1)
        
        writer.release()
