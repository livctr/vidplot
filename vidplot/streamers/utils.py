from typing import Any, Callable, Optional, Tuple


def _stream_with_last_frame_handling(
    target_time: float,
    prev_ts: Optional[float],
    prev_frame: Any,
    cur_ts: Optional[float],
    cur_frame: Any,
    last_frame_time: Optional[float],
    last_frame: Any,
    sample_rate: float,
    seek_func: Callable[[], Tuple[float, Any]],
    selection_method: str = "nearest"
) -> Tuple[Any, float, Any, float, Any, float, Any]:
    """
    Generic streaming logic that handles last frame continuation.
    
    Args:
        target_time: The target time to seek to
        prev_ts: Previous timestamp
        prev_frame: Previous frame/data
        cur_ts: Current timestamp
        cur_frame: Current frame/data
        last_frame_time: Last frame timestamp (for continuation)
        last_frame: Last frame data (for continuation)
        sample_rate: Sample rate for continuation logic
        seek_func: Function to get next timestamp and frame/data
        selection_method: How to select between prev and cur ("nearest", "nearest_left", "nearest_right")
    
    Returns:
        (frame, prev_ts, prev_frame, cur_ts, cur_frame, last_frame_time, last_frame)
    """
    # initialize window on first call
    if prev_ts is None:
        # first frame
        prev_ts, prev_frame = seek_func()
        # attempt second frame; if unavailable, duplicate prev for cur
        try:
            cur_ts, cur_frame = seek_func()
        except StopIteration:
            cur_ts, cur_frame = prev_ts, prev_frame

    # advance window until cur_ts >= target_time
    while cur_ts < target_time:
        prev_ts, prev_frame = cur_ts, cur_frame
        try:
            cur_ts, cur_frame = seek_func()
        except StopIteration:
            # Video ended, but we should continue until we've processed the last frame
            # at the target sample rate
            if last_frame_time is None:
                # Cache the last frame we have
                last_frame_time = cur_ts
                last_frame = cur_frame
            
            # Continue rendering the last frame until target_time exceeds last_frame_time + fps
            fps = 1.0 / sample_rate
            if target_time > last_frame_time + fps:
                raise StopIteration
            
            # Return the last frame for any remaining time points
            return last_frame, prev_ts, prev_frame, cur_ts, cur_frame, last_frame_time, last_frame

    # choose frame based on selection method
    if selection_method == "nearest":
        if abs(prev_ts - target_time) <= abs(cur_ts - target_time):
            return prev_frame, prev_ts, prev_frame, cur_ts, cur_frame, last_frame_time, last_frame
        return cur_frame, prev_ts, prev_frame, cur_ts, cur_frame, last_frame_time, last_frame
    elif selection_method == "nearest_left":
        return prev_frame, prev_ts, prev_frame, cur_ts, cur_frame, last_frame_time, last_frame
    elif selection_method == "nearest_right":
        return cur_frame, prev_ts, prev_frame, cur_ts, cur_frame, last_frame_time, last_frame
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")
