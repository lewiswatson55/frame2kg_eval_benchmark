"""Ground truth data adapters for HuggingFace and local datasets."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
from frame2kg_eval.io.schema import standardize_graph, validate_graph
from frame2kg_eval.utils.ids import parse_filename


class GroundTruthAdapter(ABC):
    """Abstract base class for ground truth data sources."""
    
    @abstractmethod
    def get_graph(self, video_id: str, frame_no: int) -> Optional[Dict]:
        """Get ground truth graph for a specific frame."""
        pass
    
    @abstractmethod
    def iter_frames(self) -> Iterator[Tuple[str, int, Dict]]:
        """Iterate over all frames in the dataset."""
        pass
    
    @abstractmethod
    def count_frames(self) -> int:
        """Get total number of frames."""
        pass


class HFDatasetAdapter(GroundTruthAdapter):
    """Adapter for HuggingFace Frame2KG dataset."""
    
    def __init__(self, dataset_name: str = "lewiswatson/Frame2KG-YC2", 
                 split: str = "validation_dev"):
        """Initialize with HuggingFace dataset.
        
        Args:
            dataset_name: Name of HuggingFace dataset
            split: Dataset split to use (validation_dev, testing, etc.)
        """
        from datasets import load_dataset
        
        self.dataset_name = dataset_name
        self.split = split
        
        # Load dataset
        self.dataset = load_dataset(dataset_name)[split]
        
        # Build index for fast lookup
        self._index = {}
        for idx, example in enumerate(self.dataset):
            video_id = example["video_id"]
            frame_no = int(example["frame_number"])
            self._index[(video_id, frame_no)] = idx
    
    def get_graph(self, video_id: str, frame_no: int) -> Optional[Dict]:
        """Get standardized ground truth graph."""
        key = (video_id, frame_no)
        if key not in self._index:
            return None
        
        idx = self._index[key]
        example = self.dataset[idx]
        
        # Handle graph field (might be JSON string or dict)
        raw_graph = example.get("graph")
        if isinstance(raw_graph, str):
            try:
                raw_graph = json.loads(raw_graph)
            except json.JSONDecodeError:
                return None
        
        if not validate_graph(raw_graph):
            return None
        
        return standardize_graph(raw_graph)
    
    def iter_frames(self) -> Iterator[Tuple[str, int, Dict]]:
        """Iterate over all frames with their graphs."""
        for example in self.dataset:
            video_id = example["video_id"]
            frame_no = int(example["frame_number"])
            graph = self.get_graph(video_id, frame_no)
            if graph is not None:
                yield video_id, frame_no, graph
    
    def count_frames(self) -> int:
        """Get total number of frames."""
        return len(self.dataset)
    
    def get_image(self, video_id: str, frame_no: int):
        """Get PIL image for a frame (optional utility)."""
        key = (video_id, frame_no)
        if key not in self._index:
            return None
        
        idx = self._index[key]
        return self.dataset[idx].get("image")


class LocalJsonAdapter(GroundTruthAdapter):
    """Adapter for local JSON ground truth files."""
    
    def __init__(self, root_dir: Path):
        """Initialize with local directory.
        
        Args:
            root_dir: Directory containing ground truth JSON files
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise ValueError(f"Ground truth directory does not exist: {root_dir}")
        
        self._index = {}
        self._build_index()
    
    def _build_index(self):
        """Build index of available ground truth files."""
        for filepath in self.root_dir.glob("*.json"):
            parsed = parse_filename(filepath.name)
            if parsed:
                video_id, frame_no = parsed
                self._index[(video_id, frame_no)] = filepath
    
    def get_graph(self, video_id: str, frame_no: int) -> Optional[Dict]:
        """Get standardized ground truth graph."""
        key = (video_id, frame_no)
        if key not in self._index:
            return None
        
        filepath = self._index[key]
        
        try:
            with open(filepath, 'r') as f:
                raw_graph = json.load(f)
            
            if not validate_graph(raw_graph):
                return None
            
            return standardize_graph(raw_graph)
        
        except (json.JSONDecodeError, IOError):
            return None
    
    def iter_frames(self) -> Iterator[Tuple[str, int, Dict]]:
        """Iterate over all frames with their graphs."""
        for (video_id, frame_no), filepath in sorted(self._index.items()):
            graph = self.get_graph(video_id, frame_no)
            if graph is not None:
                yield video_id, frame_no, graph
    
    def count_frames(self) -> int:
        """Get total number of frames."""
        return len(self._index)


def create_ground_truth_adapter(spec: str) -> GroundTruthAdapter:
    """Factory function to create appropriate ground truth adapter.
    
    Args:
        spec: Ground truth specification
              - "hf:dataset:split" for HuggingFace datasets
              - Path for local JSON directory
    
    Returns:
        Appropriate GroundTruthAdapter instance
    """
    if spec.startswith("hf:"):
        # Parse HuggingFace specification
        parts = spec[3:].split(":")
        if len(parts) == 2:
            dataset_name, split = parts
            return HFDatasetAdapter(dataset_name, split)
        elif len(parts) == 1:
            # Default split
            return HFDatasetAdapter(parts[0], "validation_dev")
        else:
            raise ValueError(f"Invalid HuggingFace spec: {spec}")
    else:
        # Treat as local path
        return LocalJsonAdapter(Path(spec))
