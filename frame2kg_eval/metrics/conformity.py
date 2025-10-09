"""Schema conformity metrics for prediction files."""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json


def check_node_schema(node: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Check if a node conforms to the expected schema.
    
    Expected schema:
    {
        "id": "string",
        "label": "string",
        "attributes": {"key": "value"},  # optional
        "location": "x1, y1, x2, y2[, confidence]"
    }
    
    Args:
        node: Node dictionary to check
    
    Returns:
        Tuple of (is_conformant, list_of_issues)
    """
    issues = []
    
    # Check required fields
    if "id" not in node:
        issues.append("missing 'id' field")
    elif not isinstance(node["id"], str):
        issues.append("'id' must be string")
    
    if "label" not in node:
        issues.append("missing 'label' field")
    elif not isinstance(node["label"], str):
        issues.append("'label' must be string")
    
    if "location" not in node:
        issues.append("missing 'location' field")
    elif not isinstance(node["location"], str):
        issues.append("'location' must be normalized string 'x1,y1,x2,y2,confidence'")
    else:
        parts = [p.strip() for p in node["location"].split(",")]
        if len(parts) != 5:
            issues.append("'location' must have exactly 5 values (x1,y1,x2,y2,conf)")
        else:
            try:
                x1, y1, x2, y2, conf = [float(p) for p in parts]
                # Range checks
                if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
                    issues.append("location coordinates must be normalized to [0,1]")
                if not (x1 < x2 and y1 < y2):
                    issues.append("location must satisfy x1<x2 and y1<y2")
                if not (0.0 <= conf <= 1.0):
                    issues.append("confidence must be in [0,1]")
            except (ValueError, TypeError):
                issues.append("'location' values must be numeric floats")
    
    # Check optional attributes field
    if "attributes" in node:
        if not isinstance(node["attributes"], dict):
            issues.append("'attributes' must be dictionary")
        else:
            # Check that all values are strings
            for key, value in node["attributes"].items():
                if not isinstance(key, str):
                    issues.append(f"attribute key must be string, got {type(key).__name__}")
                    break
                if not isinstance(value, (str, int, float, bool)):
                    issues.append(f"attribute value for '{key}' must be primitive type")
    
    return len(issues) == 0, issues


def check_edge_schema(edge: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Check if an edge conforms to the expected schema.
    
    Expected schema:
    {
        "source": "nodeId",
        "predicate": "string",
        "target": "nodeId"
    }
    
    Args:
        edge: Edge dictionary to check
    
    Returns:
        Tuple of (is_conformant, list_of_issues)
    """
    issues = []
    
    # Check required fields
    if "source" not in edge:
        issues.append("missing 'source' field")
    elif not isinstance(edge["source"], str):
        issues.append("'source' must be string")
    
    if "predicate" not in edge:
        issues.append("missing 'predicate' field")
    elif not isinstance(edge["predicate"], str):
        issues.append("'predicate' must be string")
    
    if "target" not in edge:
        issues.append("missing 'target' field")
    elif not isinstance(edge["target"], str):
        issues.append("'target' must be string")
    
    return len(issues) == 0, issues


def check_graph_schema(graph: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
    """Check if a graph conforms to the expected schema.
    
    Args:
        graph: Graph dictionary to check
    
    Returns:
        Tuple of (is_conformant, detailed_report)
    """
    report = {
        "conformant": False,
        "has_nodes": False,
        "has_edges": False,
        "nodes_conformant": 0,
        "nodes_total": 0,
        "edges_conformant": 0,
        "edges_total": 0,
        "issues": []
    }
    
    # Check top-level structure
    if not isinstance(graph, dict):
        report["issues"].append("graph must be dictionary")
        return False, report
    
    # Check nodes
    if "nodes" not in graph:
        report["issues"].append("missing 'nodes' field")
    else:
        report["has_nodes"] = True
        nodes = graph["nodes"]
        
        if not isinstance(nodes, list):
            report["issues"].append("'nodes' must be list")
        else:
            report["nodes_total"] = len(nodes)
            for i, node in enumerate(nodes):
                if not isinstance(node, dict):
                    report["issues"].append(f"node[{i}] must be dictionary")
                    continue
                
                is_conformant, node_issues = check_node_schema(node)
                if is_conformant:
                    report["nodes_conformant"] += 1
                else:
                    for issue in node_issues:
                        report["issues"].append(f"node[{i}]: {issue}")
    
    # Check edges
    if "edges" not in graph:
        report["issues"].append("missing 'edges' field")
    else:
        report["has_edges"] = True
        edges = graph["edges"]
        
        if not isinstance(edges, list):
            report["issues"].append("'edges' must be list")
        else:
            report["edges_total"] = len(edges)
            for i, edge in enumerate(edges):
                if not isinstance(edge, dict):
                    report["issues"].append(f"edge[{i}] must be dictionary")
                    continue
                
                is_conformant, edge_issues = check_edge_schema(edge)
                if is_conformant:
                    report["edges_conformant"] += 1
                else:
                    for issue in edge_issues:
                        report["issues"].append(f"edge[{i}]: {issue}")
    
    # Overall conformance: must have both fields and all items conformant
    report["conformant"] = (
        report["has_nodes"] and 
        report["has_edges"] and 
        report["nodes_conformant"] == report["nodes_total"] and
        report["edges_conformant"] == report["edges_total"] and
        len(report["issues"]) == 0
    )
    
    return report["conformant"], report


def check_file_conformity(filepath: Path) -> tuple[bool, Optional[Dict[str, Any]]]:
    """Check if a file contains schema-conformant JSON.
    
    Args:
        filepath: Path to file to check
    
    Returns:
        Tuple of (is_conformant, detailed_report or None if not valid JSON)
    """
    # Raw text files are not conformant
    if filepath.suffix == ".txt" or filepath.name.endswith(".raw.txt"):
        return False, None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return check_graph_schema(data)
    
    except (json.JSONDecodeError, IOError):
        # Not valid JSON, so can't check schema
        return False, None


def schema_conformity(file_records: List[Dict]) -> Dict:
    """Compute schema conformity statistics.
    
    Args:
        file_records: List of file records with conformity information
    
    Returns:
        Dictionary with conformity statistics
    """
    conformant_count = 0
    non_conformant_count = 0
    invalid_json_count = 0
    
    for record in file_records:
        if record.get("valid_json", False):
            if record.get("conformant", False):
                conformant_count += 1
            else:
                non_conformant_count += 1
        else:
            invalid_json_count += 1
    
    total_count = conformant_count + non_conformant_count + invalid_json_count
    valid_json_count = conformant_count + non_conformant_count
    
    # Conformity rate among valid JSON files
    conformity_rate_valid = (
        (conformant_count / valid_json_count * 100) 
        if valid_json_count > 0 else 0.0
    )
    
    # Overall conformity rate
    conformity_rate_total = (
        (conformant_count / total_count * 100) 
        if total_count > 0 else 0.0
    )
    
    return {
        "conformant_count": conformant_count,
        "non_conformant_count": non_conformant_count,
        "invalid_json_count": invalid_json_count,
        "total_count": total_count,
        "conformity_rate_valid_json": conformity_rate_valid,
        "conformity_rate_total": conformity_rate_total
    }


def compute_conformity_from_directory(pred_dir: Path) -> Dict:
    """Compute schema conformity statistics for all files in a directory.
    
    Args:
        pred_dir: Directory containing prediction files
    
    Returns:
        Conformity statistics dictionary with detailed breakdown
    """
    import json as json_module
    from frame2kg_eval.utils.ids import parse_filename
    
    file_records = []
    conformity_issues = {}
    
    # Check JSON files
    for filepath in pred_dir.glob("*.json"):
        parsed = parse_filename(filepath.name)
        if parsed:
            # Check if it's valid JSON first
            try:
                with open(filepath, 'r') as f:
                    json_module.load(f)
                valid_json = True
            except (json_module.JSONDecodeError, IOError):
                valid_json = False
            
            # Check conformity
            conformant, report = check_file_conformity(filepath)
            
            record = {
                "path": filepath,
                "valid_json": valid_json,
                "conformant": conformant,
                "video_id": parsed[0],
                "frame_no": parsed[1]
            }
            file_records.append(record)
            
            # Track issues for non-conformant files
            if valid_json and not conformant and report:
                key = f"{parsed[0]}.{parsed[1]}"
                conformity_issues[key] = report["issues"][:5]  # Limit to first 5 issues
    
    # Check raw text files (always non-conformant)
    for filepath in pred_dir.glob("*.raw.txt"):
        parsed = parse_filename(filepath.name)
        if parsed:
            file_records.append({
                "path": filepath,
                "valid_json": False,
                "conformant": False,
                "video_id": parsed[0],
                "frame_no": parsed[1]
            })
    
    stats = schema_conformity(file_records)
    stats["sample_issues"] = conformity_issues
    
    return stats
