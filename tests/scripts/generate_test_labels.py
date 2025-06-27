import numpy as np
import pandas as pd
import json
from pathlib import Path


def generate_labels_for_video(video_path, label_dir, label_name="label"):
    """
    Generate frame-level labels for a video file and save in multiple formats.
    Args:
        video_path: Path to the video file
        label_dir: Directory to save label files
        label_name: Name of the label column/key
    """
    import cv2

    video_path = Path(video_path)
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    times = np.arange(n_frames) / fps
    # Simple label: alternate between 0 and 1 every second
    labels = [(int(t) % 2) for t in times]
    # DataFrame
    df = pd.DataFrame({"time": times, label_name: labels})
    # Save as CSV
    csv_path = label_dir / f"{video_path.stem}.csv"
    df.to_csv(csv_path, index=False)
    # Save as JSON
    json_path = label_dir / f"{video_path.stem}.json"
    with open(json_path, "w") as f:
        json.dump({"time": times.tolist(), label_name: labels}, f)
    # Save as NPZ
    npz_path = label_dir / f"{video_path.stem}.npz"
    np.savez(npz_path, time=times, label=labels)
    # Save as dict (for in-memory tests)
    dict_path = label_dir / f"{video_path.stem}_dict.json"
    with open(dict_path, "w") as f:
        json.dump({"time": times.tolist(), label_name: labels}, f)
    print(f"Labels for {video_path.name} saved to {label_dir}")


def main():
    test_data_dir = Path("tests/input/videos")
    label_dir = Path("tests/input/frame_labels")
    label_dir.mkdir(exist_ok=True)
    # Find all mp4 videos in test_data_dir
    for video_path in test_data_dir.glob("*.mp4"):
        generate_labels_for_video(video_path, label_dir)


if __name__ == "__main__":
    main()
