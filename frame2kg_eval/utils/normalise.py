"""Text and data normalization utilities."""

import re
from typing import Any, List, Optional, Tuple, Union


def normalise_label(label: Any) -> str:
    """Normalize a label string for comparison.
    
    - Convert to lowercase
    - Replace underscores/hyphens with spaces
    - Remove special characters
    - Collapse whitespace
    """
    text = str(label) if label is not None else ""
    text = text.lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalise_id(node_id: Any) -> str:
    """Normalize a node ID for text similarity.
    
    Removes digits to focus on semantic content.
    """
    text = str(node_id) if node_id is not None else ""
    text = re.sub(r"\d+", "", text)
    return normalise_label(text)


def parse_location(loc: Union[str, List, None]) -> Optional[Tuple[float, float, float, float]]:
    """Parse location into normalized bbox coordinates.
    
    Args:
        loc: Location as string "x1,y1,x2,y2[,conf]" or list [x1,y1,x2,y2]
    
    Returns:
        Tuple of (x1, y1, x2, y2) clamped to [0,1], or None if invalid
    """
    if loc is None:
        return None
    
    try:
        if isinstance(loc, str):
            parts = [float(x) for x in loc.split(",")]
        elif isinstance(loc, (list, tuple)):
            parts = [float(x) for x in loc]
        else:
            return None
        
        if len(parts) < 4:
            return None
        
        x1, y1, x2, y2 = parts[:4]
        
        # Ensure proper ordering (top-left to bottom-right) first
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Then clamp to [0, 1]
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        
        # Avoid zero-area boxes
        if x1 == x2:
            x2 = min(1.0, x1 + 1e-6)
        if y1 == y2:
            y2 = min(1.0, y1 + 1e-6)
        
        return (x1, y1, x2, y2)
    
    except (ValueError, TypeError, AttributeError):
        return None


def extract_confidence(loc: Union[str, List, None]) -> Optional[float]:
    """Extract confidence score from location string if present.
    
    Args:
        loc: Location string "x1,y1,x2,y2,conf" or list
    
    Returns:
        Confidence score or None
    """
    if loc is None:
        return None
    
    try:
        if isinstance(loc, str):
            parts = [float(x) for x in loc.split(",")]
        elif isinstance(loc, (list, tuple)):
            parts = [float(x) for x in loc]
        else:
            return None
        
        if len(parts) >= 5:
            return float(parts[4])
        
    except (ValueError, TypeError, AttributeError):
        pass
    
    return None


def normalise_predicate(pred: str) -> str:
    """Normalize edge predicate for comparison."""
    return normalise_label(pred)
