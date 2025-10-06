"""Schema standardization and validation for graph data."""

from typing import Any, Dict, List, Optional
from frame2kg_eval.utils.normalise import parse_location, extract_confidence


def standardize_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize a node to consistent schema.
    
    Expected output format:
    {
        "id": str,
        "label": str,
        "attributes": dict,
        "location": [x1, y1, x2, y2],
        "conf": float or None
    }
    """
    # Handle location parsing
    raw_location = node.get("location")
    bbox = parse_location(raw_location)
    conf = extract_confidence(raw_location)
    
    # Standardize attributes
    attrs = node.get("attributes", {})
    if not isinstance(attrs, dict):
        attrs = {}
    
    return {
        "id": str(node.get("id", "")),
        "label": str(node.get("label", "")),
        "attributes": attrs,
        "location": list(bbox) if bbox else None,
        "conf": conf
    }


def standardize_edge(edge: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize an edge to consistent schema.
    
    Expected output format:
    {
        "source": str,
        "target": str, 
        "predicate": str
    }
    """
    return {
        "source": str(edge.get("source", "")),
        "target": str(edge.get("target", "")),
        "predicate": str(edge.get("predicate", ""))
    }


def standardize_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize a complete graph to consistent schema.
    
    Args:
        graph: Raw graph dictionary
        
    Returns:
        Standardized graph with nodes and edges lists
    """
    # Handle nodes
    raw_nodes = graph.get("nodes", [])
    if not isinstance(raw_nodes, list):
        raw_nodes = []
    
    nodes = []
    for node in raw_nodes:
        if isinstance(node, dict):
            try:
                std_node = standardize_node(node)
                nodes.append(std_node)
            except Exception:
                # Skip malformed nodes
                continue
    
    # Handle edges
    raw_edges = graph.get("edges", [])
    if not isinstance(raw_edges, list):
        raw_edges = []
    
    edges = []
    for edge in raw_edges:
        if isinstance(edge, dict):
            try:
                # Check if edge has required fields
                if not all(k in edge for k in ["source", "target", "predicate"]):
                    continue
                std_edge = standardize_edge(edge)
                edges.append(std_edge)
            except Exception:
                # Skip malformed edges
                continue
    
    return {
        "nodes": nodes,
        "edges": edges
    }


def validate_graph(graph: Any) -> bool:
    """Check if a graph object is valid.
    
    Args:
        graph: Object to validate
        
    Returns:
        True if valid graph structure, False otherwise
    """
    if not isinstance(graph, dict):
        return False
    
    if "nodes" not in graph and "edges" not in graph:
        return False
    
    # Check nodes structure
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        return False
    
    for node in nodes:
        if not isinstance(node, dict):
            return False
        if "id" not in node:
            return False
    
    # Check edges structure
    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        return False
    
    for edge in edges:
        if not isinstance(edge, dict):
            return False
        if not all(k in edge for k in ["source", "target", "predicate"]):
            return False
    
    return True
