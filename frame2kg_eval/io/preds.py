"""Prediction file loading and indexing."""

import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
from frame2kg_eval.utils.ids import parse_filename
from frame2kg_eval.io.schema import standardize_graph, validate_graph


class PredictionLoader:
    """Load and index prediction files from a directory."""
    
    def __init__(self, pred_dir: Path):
        """Initialize loader with prediction directory.
        
        Args:
            pred_dir: Path to directory containing prediction files
        """
        self.pred_dir = Path(pred_dir)
        if not self.pred_dir.exists():
            raise ValueError(f"Prediction directory does not exist: {pred_dir}")
        
        self._index = {}
        self._build_index()
    
    def _build_index(self):
        """Build index of available prediction files."""
        json_files = list(self.pred_dir.glob("*.json"))
        raw_files = list(self.pred_dir.glob("*.raw.txt"))
        
        for filepath in json_files + raw_files:
            parsed = parse_filename(filepath.name)
            if parsed:
                video_id, frame_no = parsed
                self._index[(video_id, frame_no)] = filepath
    
    def get_graph(self, video_id: str, frame_no: int) -> Optional[Dict]:
        """Get standardized graph for a specific frame.
        
        Args:
            video_id: Video identifier
            frame_no: Frame number
            
        Returns:
            Standardized graph dict or None if not found/invalid
        """
        key = (video_id, frame_no)
        if key not in self._index:
            return None
        
        filepath = self._index[key]
        
        # Skip .raw.txt files as they're invalid JSON
        if filepath.suffix == ".txt":
            return None
        
        try:
            with open(filepath, 'r') as f:
                raw_graph = json.load(f)
            
            if not validate_graph(raw_graph):
                return None
            
            return standardize_graph(raw_graph)
        
        except (json.JSONDecodeError, IOError):
            return None
    
    def iter_predictions(self) -> Iterator[Tuple[str, int, Optional[Dict], Path]]:
        """Iterate over all predictions.
        
        Yields:
            Tuples of (video_id, frame_no, graph, filepath)
        """
        for (video_id, frame_no), filepath in sorted(self._index.items()):
            graph = self.get_graph(video_id, frame_no)
            yield video_id, frame_no, graph, filepath
    
    def get_index(self) -> Dict[Tuple[str, int], Path]:
        """Get the full file index.
        
        Returns:
            Dict mapping (video_id, frame_no) to file paths
        """
        return self._index.copy()
    
    def count_valid(self) -> Tuple[int, int, int]:
        """Count valid, invalid, and total files.
        
        Returns:
            Tuple of (valid_count, invalid_count, total_count)
        """
        valid = 0
        invalid = 0
        
        for (vid, fno), path in self._index.items():
            if path.suffix == ".txt":
                invalid += 1
            else:
                # Try to load and validate
                graph = self.get_graph(vid, fno)
                if graph is not None:
                    valid += 1
                else:
                    invalid += 1
        
        return valid, invalid, valid + invalid


def index_predictions(pred_dir: Path) -> Dict[Tuple[str, int], Path]:
    """Build an index of prediction files in a directory.
    
    Args:
        pred_dir: Directory containing prediction files
        
    Returns:
        Dict mapping (video_id, frame_no) to file paths
    """
    loader = PredictionLoader(pred_dir)
    return loader.get_index()
