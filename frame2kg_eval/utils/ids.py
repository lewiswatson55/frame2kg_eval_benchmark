"""File naming and ID parsing utilities."""

import re
from pathlib import Path
from typing import Optional, Tuple


def parse_filename(filename: str) -> Optional[Tuple[str, int]]:
    """Parse video_id and frame_number from filename.
    
    Expected format: <video_id>.<frame_number>.json or .raw.txt
    
    Args:
        filename: Name of the file (with or without path)
    
    Returns:
        Tuple of (video_id, frame_number) or None if invalid
    """
    # Get just the filename if it's a path
    if isinstance(filename, Path):
        filename = filename.name
    else:
        filename = Path(filename).name
    
    # Pattern: <video_id>.<frame_no>.(json|raw.txt)
    pattern = r"^(.+?)\.(\d+)\.(json|raw\.txt)$"
    match = re.match(pattern, filename)
    
    if match:
        video_id = match.group(1)
        frame_no = int(match.group(2))
        return (video_id, frame_no)
    
    return None


def build_filename(video_id: str, frame_no: int, is_json: bool = True) -> str:
    """Construct standard filename from video_id and frame_number.
    
    Args:
        video_id: Video identifier
        frame_no: Frame number
        is_json: If True, returns .json extension, else .raw.txt
    
    Returns:
        Formatted filename
    """
    ext = "json" if is_json else "raw.txt"
    return f"{video_id}.{frame_no}.{ext}"


def resolve_prediction_path(
    video_id: str, 
    frame_no: int, 
    pred_index: dict,
    allow_off_by_one: bool = True
) -> Optional[Path]:
    """Resolve prediction file path with fallback strategies.
    
    Args:
        video_id: Video identifier
        frame_no: Frame number
        pred_index: Dict mapping (video_id, frame_no) -> Path
        allow_off_by_one: Whether to try frame_no ± 1 as fallback
    
    Returns:
        Path to prediction file or None if not found
    """
    # Try exact match
    key = (video_id, frame_no)
    if key in pred_index:
        return pred_index[key]
    
    # Try off-by-one if allowed
    if allow_off_by_one:
        for offset in [1, -1]:
            alt_key = (video_id, frame_no + offset)
            if alt_key in pred_index:
                return pred_index[alt_key]
    
    # Try string frame number as last resort
    for (vid, fno), path in pred_index.items():
        if vid == video_id and str(fno) == str(frame_no):
            return path
    
    return None
