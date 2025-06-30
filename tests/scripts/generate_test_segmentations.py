"""
Generates and encodes simulated segmentation data for testing purposes.

This script mimics the output of a segmentation model running on a video.
It produces a series of timestamps and corresponding encoded segmentation masks
for multiple objects, saving the result to a JSON file.

The first frame's segmentation is encoded with full metadata (shape, dtype, etc.),
while subsequent frames are encoded without this redundant information to simulate
an efficient streaming format.
"""

import numpy as np
import json
import os


from vidplot.encode.segmentations import encode_segmentation_masks

# --- Simulation Parameters ---
VIDEO_HEIGHT = 480
VIDEO_WIDTH = 640
NUM_FRAMES = 150  # 5 seconds at 30 FPS
FPS = 30.0
MAX_OBJECTS_PER_FRAME = 3  # Let's have 3 moving objects
OUTPUT_FILEPATH = "./tests/input/frame_segmentations/segmentations.json"


class MovingObject:
    """Represents a single object moving within the video frame."""

    def __init__(self, object_id: int, frame_height: int, frame_width: int):
        self.object_id = object_id
        self.frame_height = frame_height
        self.frame_width = frame_width

        # Random initial properties
        self.w = np.random.randint(50, 120)
        self.h = np.random.randint(50, 120)
        self.x = np.random.randint(0, self.frame_width - self.w)
        self.y = np.random.randint(0, self.frame_height - self.h)

        # Slow, random velocity
        self.dx = np.random.uniform(-2.5, 2.5)
        self.dy = np.random.uniform(-2.5, 2.5)

    def update(self):
        """Updates the object's position and handles bouncing off edges."""
        self.x += self.dx
        self.y += self.dy

        # Bounce off horizontal walls
        if self.x < 0:
            self.x = 0
            self.dx *= -1
        elif self.x + self.w > self.frame_width:
            self.x = self.frame_width - self.w
            self.dx *= -1

        # Bounce off vertical walls
        if self.y < 0:
            self.y = 0
            self.dy *= -1
        elif self.y + self.h > self.frame_height:
            self.y = self.frame_height - self.h
            self.dy *= -1

    def get_mask(self) -> np.ndarray:
        """Generates a binary mask for the object's current position."""
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)

        # Get integer coordinates for the rectangle
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = int(self.x + self.w), int(self.y + self.h)

        # Clip coordinates to be safely within frame bounds
        x1_c = np.clip(x1, 0, self.frame_width)
        y1_c = np.clip(y1, 0, self.frame_height)
        x2_c = np.clip(x2, 0, self.frame_width)
        y2_c = np.clip(y2, 0, self.frame_height)

        mask[y1_c:y2_c, x1_c:x2_c] = 1
        return mask


def simulate_segmentation_encoding():
    """
    Simulates the process of receiving and encoding segmentation data from a video.

    This version uses persistent objects that move around the frame.
    """
    frame_times = []
    encoded_outputs = []

    print(f"Simulating {NUM_FRAMES} frames with {MAX_OBJECTS_PER_FRAME} moving objects...")

    # Create persistent moving objects
    moving_objects = [
        MovingObject(obj_id, VIDEO_HEIGHT, VIDEO_WIDTH)
        for obj_id in range(1, MAX_OBJECTS_PER_FRAME + 1)
    ]

    for i in range(NUM_FRAMES):
        timestamp = i / FPS
        frame_times.append(timestamp)

        seg_ids = []
        seg_masks = []

        # Update and get masks for all objects for the current frame
        for obj in moving_objects:
            obj.update()
            seg_ids.append(obj.object_id)
            seg_masks.append(obj.get_mask())

        # The first frame includes metadata; subsequent frames do not
        is_first_frame = i == 0

        encoded_data = encode_segmentation_masks(
            seg_ids=seg_ids, seg_masks=seg_masks, save_metadata=is_first_frame
        )

        encoded_outputs.append(encoded_data)

        if is_first_frame:
            print("Encoded first frame with full metadata.")

    print(f"Simulation complete. Processed {len(frame_times)} frames.")

    return {"frame_times": frame_times, "encoded_outputs": encoded_outputs}


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILEPATH), exist_ok=True)

    # Generate the simulated data
    simulation_result = simulate_segmentation_encoding()

    print(f"Saving simulated data to: {OUTPUT_FILEPATH}")

    # Save to JSON without indentation for a more compact file
    with open(OUTPUT_FILEPATH, "w") as f:
        json.dump(simulation_result, f)

    print("Successfully saved test segmentation data.")
