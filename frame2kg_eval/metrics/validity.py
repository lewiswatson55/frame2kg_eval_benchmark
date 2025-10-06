"""JSON validity metrics for prediction files."""

from pathlib import Path
from typing import Dict, List


def json_validity(file_records: List[Dict]) -> Dict:
    """Compute JSON validity statistics.
    
    Args:
        file_records: List of file records with 'path' and 'valid' fields
    
    Returns:
        Dictionary with validity statistics:
            - valid_count: Number of valid JSON files
            - invalid_count: Number of invalid/raw files
            - total_count: Total number of files
            - validity_rate: Percentage of valid files
    """
    valid_count = 0
    invalid_count = 0
    
    for record in file_records:
        if record.get("valid", False):
            valid_count += 1
        else:
            invalid_count += 1
    
    total_count = valid_count + invalid_count
    validity_rate = (valid_count / total_count * 100) if total_count > 0 else 0.0
    
    return {
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "total_count": total_count,
        "validity_rate": validity_rate
    }


def check_file_validity(filepath: Path) -> bool:
    """Check if a file contains valid JSON.
    
    Args:
        filepath: Path to file to check
    
    Returns:
        True if file contains valid JSON with required structure
    """
    import json
    from frame2kg_eval.io.schema import validate_graph
    
    # Raw text files are invalid
    if filepath.suffix == ".txt" or filepath.name.endswith(".raw.txt"):
        return False
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check if it's a valid graph structure
        return validate_graph(data)
    
    except (json.JSONDecodeError, IOError, Exception):
        return False


def compute_validity_from_directory(pred_dir: Path) -> Dict:
    """Compute validity statistics for all files in a directory.
    
    Args:
        pred_dir: Directory containing prediction files
    
    Returns:
        Validity statistics dictionary
    """
    from frame2kg_eval.utils.ids import parse_filename
    
    file_records = []
    
    # Check JSON files
    for filepath in pred_dir.glob("*.json"):
        parsed = parse_filename(filepath.name)
        if parsed:
            valid = check_file_validity(filepath)
            file_records.append({
                "path": filepath,
                "valid": valid,
                "video_id": parsed[0],
                "frame_no": parsed[1]
            })
    
    # Check raw text files
    for filepath in pred_dir.glob("*.raw.txt"):
        parsed = parse_filename(filepath.name)
        if parsed:
            file_records.append({
                "path": filepath,
                "valid": False,  # Raw files are always invalid
                "video_id": parsed[0],
                "frame_no": parsed[1]
            })
    
    return json_validity(file_records)
