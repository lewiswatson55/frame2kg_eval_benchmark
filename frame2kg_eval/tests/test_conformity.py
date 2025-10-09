"""Tests for schema conformity metrics."""

import json
import tempfile
from pathlib import Path
import pytest

from frame2kg_eval.metrics.conformity import (
    check_node_schema,
    check_edge_schema,
    check_graph_schema,
    check_file_conformity,
    compute_conformity_from_directory
)


class TestNodeSchema:
    """Test node schema validation."""
    
    def test_valid_node_minimal(self):
        """Test minimal valid node."""
        node = {
            "id": "n1",
            "label": "person",
            "location": "0.1,0.2,0.3,0.4,0.9"
        }
        is_valid, issues = check_node_schema(node)
        assert is_valid is True
        assert len(issues) == 0
    
    def test_valid_node_with_attributes(self):
        """Test valid node with attributes."""
        node = {
            "id": "n1",
            "label": "person",
            "location": "0.1,0.2,0.3,0.4,0.95",
            "attributes": {"color": "blue", "size": "large"}
        }
        is_valid, issues = check_node_schema(node)
        assert is_valid is True
        assert len(issues) == 0
    
    def test_location_list_disallowed(self):
        """Test node with location as list is not conformant."""
        node = {
            "id": "n1",
            "label": "person",
            "location": [0.1, 0.2, 0.3, 0.4, 0.9]
        }
        is_valid, issues = check_node_schema(node)
        assert is_valid is False
        assert any("must be normalized string" in msg for msg in issues)
    
    def test_missing_required_fields(self):
        """Test node missing required fields."""
        # Missing id
        node = {"label": "person", "location": "0.1,0.2,0.3,0.4,0.9"}
        is_valid, issues = check_node_schema(node)
        assert is_valid is False
        assert "missing 'id' field" in issues
        
        # Missing label
        node = {"id": "n1", "location": "0.1,0.2,0.3,0.4,0.9"}
        is_valid, issues = check_node_schema(node)
        assert is_valid is False
        assert "missing 'label' field" in issues
        
        # Missing location
        node = {"id": "n1", "label": "person"}
        is_valid, issues = check_node_schema(node)
        assert is_valid is False
        assert "missing 'location' field" in issues
    
    def test_invalid_types(self):
        """Test node with invalid field types."""
        # Non-string id
        node = {"id": 123, "label": "person", "location": "0.1,0.2,0.3,0.4,0.9"}
        is_valid, issues = check_node_schema(node)
        assert is_valid is False
        assert "'id' must be string" in issues
        
        # Non-dict attributes
        node = {
            "id": "n1",
            "label": "person",
            "location": "0.1,0.2,0.3,0.4,0.9",
            "attributes": "not a dict"
        }
        is_valid, issues = check_node_schema(node)
        assert is_valid is False
        assert "'attributes' must be dictionary" in issues
    
    def test_invalid_location_format(self):
        """Test invalid location format."""
        # Too few values in string
        node = {"id": "n1", "label": "person", "location": "0.1,0.2,0.3,0.4"}
        is_valid, issues = check_node_schema(node)
        assert is_valid is False
        assert any("must have exactly 5 values" in msg for msg in issues)
        
        # Disallow list format
        node = {"id": "n1", "label": "person", "location": [0.1, 0.2, 0.3, 0.4]}
        is_valid, issues = check_node_schema(node)
        assert is_valid is False
        assert any("must be normalized string" in msg for msg in issues)


class TestEdgeSchema:
    """Test edge schema validation."""
    
    def test_valid_edge(self):
        """Test valid edge."""
        edge = {
            "source": "n1",
            "target": "n2",
            "predicate": "holding"
        }
        is_valid, issues = check_edge_schema(edge)
        assert is_valid is True
        assert len(issues) == 0
    
    def test_missing_fields(self):
        """Test edge missing required fields."""
        # Missing source
        edge = {"target": "n2", "predicate": "holding"}
        is_valid, issues = check_edge_schema(edge)
        assert is_valid is False
        assert "missing 'source' field" in issues
        
        # Missing predicate
        edge = {"source": "n1", "target": "n2"}
        is_valid, issues = check_edge_schema(edge)
        assert is_valid is False
        assert "missing 'predicate' field" in issues
    
    def test_invalid_types(self):
        """Test edge with invalid field types."""
        edge = {
            "source": 123,  # Should be string
            "target": "n2",
            "predicate": "holding"
        }
        is_valid, issues = check_edge_schema(edge)
        assert is_valid is False
        assert "'source' must be string" in issues


class TestGraphSchema:
    """Test complete graph schema validation."""
    
    def test_valid_graph(self):
        """Test valid complete graph."""
        graph = {
            "nodes": [
                {"id": "n1", "label": "person", "location": "0.1,0.2,0.3,0.4,0.9"},
                {"id": "n2", "label": "ball", "location": "0.5,0.6,0.7,0.8,0.8"}
            ],
            "edges": [
                {"source": "n1", "target": "n2", "predicate": "holding"}
            ]
        }
        is_valid, report = check_graph_schema(graph)
        assert is_valid is True
        assert report["conformant"] is True
        assert report["nodes_conformant"] == 2
        assert report["edges_conformant"] == 1
        assert len(report["issues"]) == 0
    
    def test_empty_graph(self):
        """Test empty but valid graph."""
        graph = {"nodes": [], "edges": []}
        is_valid, report = check_graph_schema(graph)
        assert is_valid is True
        assert report["conformant"] is True
    
    def test_missing_top_level_fields(self):
        """Test graph missing top-level fields."""
        # Missing nodes
        graph = {"edges": []}
        is_valid, report = check_graph_schema(graph)
        assert is_valid is False
        assert "missing 'nodes' field" in report["issues"]
        
        # Missing edges
        graph = {"nodes": []}
        is_valid, report = check_graph_schema(graph)
        assert is_valid is False
        assert "missing 'edges' field" in report["issues"]
    
    def test_partial_conformance(self):
        """Test graph with partial conformance."""
        graph = {
            "nodes": [
                {"id": "n1", "label": "person", "location": "0.1,0.2,0.3,0.4,0.9"},  # Valid
                {"id": "n2", "location": "0.5,0.6,0.7,0.8,0.6"}  # Missing label
            ],
            "edges": [
                {"source": "n1", "target": "n2", "predicate": "holding"},  # Valid
                {"source": "n1", "predicate": "seeing"}  # Missing target
            ]
        }
        is_valid, report = check_graph_schema(graph)
        assert is_valid is False
        assert report["nodes_conformant"] == 1
        assert report["nodes_total"] == 2
        assert report["edges_conformant"] == 1
        assert report["edges_total"] == 2
        assert len(report["issues"]) > 0
    
    def test_invalid_structure(self):
        """Test non-dict graph."""
        graph = []
        is_valid, report = check_graph_schema(graph)
        assert is_valid is False
        assert "graph must be dictionary" in report["issues"]
    
    def test_invalid_list_types(self):
        """Test graph with non-list nodes/edges."""
        graph = {
            "nodes": "not a list",
            "edges": {}
        }
        is_valid, report = check_graph_schema(graph)
        assert is_valid is False
        assert "'nodes' must be list" in report["issues"]
        assert "'edges' must be list" in report["issues"]


class TestFileConformity:
    """Test file-level conformity checking."""
    
    def test_conformant_json_file(self):
        """Test conformant JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            with open(filepath, 'w') as f:
                json.dump({
                    "nodes": [
                        {"id": "n1", "label": "person", "location": "0.1,0.2,0.3,0.4,0.9"}
                    ],
                    "edges": []
                }, f)
            
            is_conformant, report = check_file_conformity(filepath)
            assert is_conformant is True
            assert report["conformant"] is True
    
    def test_non_conformant_json_file(self):
        """Test non-conformant JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            with open(filepath, 'w') as f:
                json.dump({
                    "nodes": [
                        {"label": "person"}  # Missing id and location
                    ],
                    "edges": []
                }, f)
            
            is_conformant, report = check_file_conformity(filepath)
            assert is_conformant is False
            assert report is not None
            assert len(report["issues"]) > 0
    
    def test_invalid_json_file(self):
        """Test invalid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            with open(filepath, 'w') as f:
                f.write("{invalid json")
            
            is_conformant, report = check_file_conformity(filepath)
            assert is_conformant is False
            assert report is None  # No report for invalid JSON
    
    def test_raw_text_file(self):
        """Test raw text file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.raw.txt"
            with open(filepath, 'w') as f:
                f.write("raw text")
            
            is_conformant, report = check_file_conformity(filepath)
            assert is_conformant is False
            assert report is None


class TestDirectoryConformity:
    """Test directory-level conformity computation."""
    
    def test_compute_conformity_from_directory(self):
        """Test conformity computation for directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Conformant JSON
            conformant_file = tmppath / "video1.001.json"
            with open(conformant_file, 'w') as f:
                json.dump({
                    "nodes": [{"id": "n1", "label": "person", "location": "0.1,0.2,0.3,0.4,0.9"}],
                    "edges": []
                }, f)
            
            # Non-conformant JSON (missing required fields)
            non_conformant_file = tmppath / "video1.002.json"
            with open(non_conformant_file, 'w') as f:
                json.dump({
                    "nodes": [{"label": "person"}],  # Missing id and location
                    "edges": []
                }, f)
            
            # Invalid JSON
            invalid_file = tmppath / "video1.003.json"
            with open(invalid_file, 'w') as f:
                f.write("{invalid json")
            
            # Raw text file
            raw_file = tmppath / "video1.004.raw.txt"
            with open(raw_file, 'w') as f:
                f.write("raw text")
            
            # Compute statistics
            stats = compute_conformity_from_directory(tmppath)
            
            assert stats["conformant_count"] == 1
            assert stats["non_conformant_count"] == 1
            assert stats["invalid_json_count"] == 2  # Invalid JSON + raw text
            assert stats["total_count"] == 4
            assert 45 < stats["conformity_rate_valid_json"] < 55  # ~50% (1/2 valid JSON)
            assert 20 < stats["conformity_rate_total"] < 30  # ~25% (1/4 total)
            
            # Check that issues are tracked
            assert "sample_issues" in stats
            assert "video1.2" in stats["sample_issues"]
            assert len(stats["sample_issues"]["video1.2"]) > 0
    
    def test_empty_directory(self):
        """Test empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            stats = compute_conformity_from_directory(tmppath)
            
            assert stats["conformant_count"] == 0
            assert stats["non_conformant_count"] == 0
            assert stats["invalid_json_count"] == 0
            assert stats["total_count"] == 0
            assert stats["conformity_rate_valid_json"] == 0.0
            assert stats["conformity_rate_total"] == 0.0
