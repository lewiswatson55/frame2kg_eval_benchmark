"""Timing metrics from manifest files."""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def manifest_timing(manifest_csv_path: Path) -> Dict:
    """Extract timing statistics from manifest CSV.
    
    Args:
        manifest_csv_path: Path to manifest.csv file
    
    Returns:
        Dictionary with timing statistics:
            - mean: Mean generation time in seconds
            - median: Median generation time
            - std: Standard deviation
            - min: Minimum time
            - max: Maximum time
            - n: Number of samples
    """
    if not manifest_csv_path.exists():
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "n": 0
        }
    
    times = []
    
    try:
        with open(manifest_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try to extract generation time
                time_str = row.get("gen_wall_time_s", "")
                if time_str:
                    try:
                        time_val = float(time_str)
                        if time_val > 0:  # Filter out invalid times
                            times.append(time_val)
                    except (ValueError, TypeError):
                        continue
    except (IOError, csv.Error):
        pass
    
    if not times:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "n": 0
        }
    
    times_array = np.array(times)
    
    return {
        "mean": float(np.mean(times_array)),
        "median": float(np.median(times_array)),
        "std": float(np.std(times_array)),
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
        "n": len(times)
    }


def extract_timing_per_frame(manifest_csv_path: Path) -> Dict[tuple, float]:
    """Extract per-frame timing from manifest.
    
    Args:
        manifest_csv_path: Path to manifest.csv
    
    Returns:
        Dictionary mapping (video_id, frame_no) to generation time
    """
    frame_times = {}
    
    if not manifest_csv_path.exists():
        return frame_times
    
    try:
        with open(manifest_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row.get("video_id", "")
                frame_no_str = row.get("frame_number", "")
                time_str = row.get("gen_wall_time_s", "")
                
                if video_id and frame_no_str and time_str:
                    try:
                        frame_no = int(frame_no_str)
                        time_val = float(time_str)
                        if time_val > 0:
                            frame_times[(video_id, frame_no)] = time_val
                    except (ValueError, TypeError):
                        continue
    except (IOError, csv.Error):
        pass
    
    return frame_times


def compute_throughput_stats(manifest_csv_path: Path) -> Dict:
    """Compute throughput statistics (frames/second).
    
    Args:
        manifest_csv_path: Path to manifest.csv
    
    Returns:
        Dictionary with throughput statistics
    """
    timing_stats = manifest_timing(manifest_csv_path)
    
    if timing_stats["n"] == 0:
        return {
            "mean_fps": None,
            "median_fps": None,
            "total_frames": 0,
            "total_time": None
        }
    
    # Compute frames per second from times
    mean_time = timing_stats["mean"]
    median_time = timing_stats["median"]
    
    mean_fps = 1.0 / mean_time if mean_time and mean_time > 0 else None
    median_fps = 1.0 / median_time if median_time and median_time > 0 else None
    
    # Total processing time (sum of all frame times)
    frame_times = extract_timing_per_frame(manifest_csv_path)
    total_time = sum(frame_times.values()) if frame_times else None
    
    return {
        "mean_fps": mean_fps,
        "median_fps": median_fps,
        "total_frames": timing_stats["n"],
        "total_time": total_time
    }
