#!/usr/bin/env python3
"""
Script to generate test videos for vidplot testing.
Generates a video with a moving colored square and saves it to test_data/.
"""

import os
import cv2
import numpy as np
from pathlib import Path

def generate_moving_square_video(output_path, duration=5.0, fps=30.0, frame_size=(640, 480)):
    """
    Generate a video with a colored square moving in a circular pattern.
    
    Args:
        output_path: Path to save the video
        duration: Duration in seconds
        fps: Frames per second
        frame_size: (width, height) of the video
    """
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    
    # Video parameters
    n_frames = int(duration * fps)
    width, height = frame_size
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 4
    square_size = 50
    
    # Colors for the square (BGR format)
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    print(f"Generating {n_frames} frames at {fps} FPS...")
    
    for frame_idx in range(n_frames):
        # Create white background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Calculate square position (circular motion)
        t = frame_idx / n_frames
        angle = 2 * np.pi * t * 2  # 2 complete rotations
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        
        # Ensure square stays within bounds
        x = max(square_size // 2, min(width - square_size // 2, x))
        y = max(square_size // 2, min(height - square_size // 2, y))
        
        # Draw square
        color_idx = int(t * len(colors)) % len(colors)
        color = colors[color_idx]
        
        x1 = x - square_size // 2
        y1 = y - square_size // 2
        x2 = x + square_size // 2
        y2 = y + square_size // 2
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        
        # Add frame number text
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add timestamp
        timestamp = frame_idx / fps
        cv2.putText(frame, f"Time: {timestamp:.2f}s", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        out.write(frame)
        
        # Progress indicator
        if frame_idx % (n_frames // 10) == 0:
            print(f"  {frame_idx}/{n_frames} frames ({100*frame_idx/n_frames:.1f}%)")
    
    out.release()
    print(f"Video saved to: {output_path}")
    print(f"Video info: {width}x{height}, {fps} FPS, {duration}s duration")

def generate_static_test_video(output_path, duration=3.0, fps=30.0, frame_size=(320, 240)):
    """
    Generate a simple static test video with a colored background and text.
    
    Args:
        output_path: Path to save the video
        duration: Duration in seconds
        fps: Frames per second
        frame_size: (width, height) of the video
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    
    n_frames = int(duration * fps)
    width, height = frame_size
    
    print(f"Generating static video: {n_frames} frames...")
    
    for frame_idx in range(n_frames):
        # Create gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a simple gradient
        for y in range(height):
            for x in range(width):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = int(255 * (x + y) / (width + height))
                frame[y, x] = [b, g, r]  # BGR format
        
        # Add text
        cv2.putText(frame, "Test Video", (width//2 - 100, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add frame info
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Static video saved to: {output_path}")

def main():
    """Generate test videos for vidplot testing."""
    # Create test_data directory
    test_data_dir = Path("tests/test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    print("Generating test videos for vidplot...")
    
    # Generate moving square video
    moving_video_path = test_data_dir / "moving_square.mp4"
    generate_moving_square_video(moving_video_path, duration=5.0, fps=30.0)
    
    # Generate static test video
    static_video_path = test_data_dir / "static_test.mp4"
    generate_static_test_video(static_video_path, duration=3.0, fps=30.0)
    
    # Generate a shorter video for quick tests
    short_video_path = test_data_dir / "short_test.mp4"
    generate_moving_square_video(short_video_path, duration=2.0, fps=15.0, frame_size=(320, 240))
    
    print("\nTest videos generated successfully!")
    print(f"Files created in: {test_data_dir.absolute()}")
    print("  - moving_square.mp4: 5s moving square video")
    print("  - static_test.mp4: 3s static gradient video") 
    print("  - short_test.mp4: 2s short test video")

if __name__ == "__main__":
    main()