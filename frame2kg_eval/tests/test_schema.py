"""Tests for schema validation and standardization."""

import pytest
from frame2kg_eval.io.schema import (
    standardize_node, standardize_edge, standardize_graph, validate_graph
)


class TestSchema:
    
    def test_standardize_node(self):
        raw_node = {
            "id": "person1",
            "label": "person",
            "location": "0.1,0.2,0.3,0.4,0.95",
            "attributes": {"appearance": "blue shirt"}
        }
        
        std_node = standardize_node(raw_node)
        
        assert std_node["id"] == "person1"
        assert std_node["label"] == "person"
        assert std_node["location"] == [0.1, 0.2, 0.3, 0.4]
        assert std_node["conf"] == 0.95
        assert std_node["attributes"]["appearance"] == "blue shirt"
    
    def test_standardize_node_minimal(self):
        raw_node = {"id": 123, "label": None}
        
        std_node = standardize_node(raw_node)
        
        assert std_node["id"] == "123"
        assert std_node["label"] == "None"
        assert std_node["location"] is None
        assert std_node["conf"] is None
        assert std_node["attributes"] == {}
    
    def test_standardize_edge(self):
        raw_edge = {
            "source": "node1",
            "target": "node2",
            "predicate": "next_to"
        }
        
        std_edge = standardize_edge(raw_edge)
        
        assert std_edge["source"] == "node1"
        assert std_edge["target"] == "node2"
        assert std_edge["predicate"] == "next_to"
    
    def test_standardize_graph(self):
        raw_graph = {
            "nodes": [
                {"id": "1", "label": "person", "location": "0.1,0.2,0.3,0.4"},
                {"id": "2", "label": "ball", "location": [0.5, 0.6, 0.7, 0.8]}
            ],
            "edges": [
                {"source": "1", "target": "2", "predicate": "holding"}
            ]
        }
        
        std_graph = standardize_graph(raw_graph)
        
        assert len(std_graph["nodes"]) == 2
        assert len(std_graph["edges"]) == 1
        assert std_graph["nodes"][0]["location"] == [0.1, 0.2, 0.3, 0.4]
        assert std_graph["nodes"][1]["location"] == [0.5, 0.6, 0.7, 0.8]
    
    def test_standardize_graph_with_invalid_items(self):
        raw_graph = {
            "nodes": [
                {"id": "1", "label": "valid"},
                "invalid_node",  # Should be skipped
                {"id": "2", "label": "also_valid"}
            ],
            "edges": [
                {"source": "1", "target": "2", "predicate": "rel"},
                {"missing": "predicate"},  # Should be skipped
            ]
        }
        
        std_graph = standardize_graph(raw_graph)
        
        assert len(std_graph["nodes"]) == 2
        assert len(std_graph["edges"]) == 1
    
    def test_validate_graph_valid(self):
        graph = {
            "nodes": [
                {"id": "1", "label": "person"}
            ],
            "edges": [
                {"source": "1", "target": "1", "predicate": "self"}
            ]
        }
        
        assert validate_graph(graph) is True
    
    def test_validate_graph_invalid(self):
        # Not a dict
        assert validate_graph([]) is False
        
        # Missing nodes and edges
        assert validate_graph({}) is False
        
        # Invalid nodes structure
        assert validate_graph({"nodes": "not_a_list", "edges": []}) is False
        
        # Node without ID
        assert validate_graph({"nodes": [{"label": "no_id"}], "edges": []}) is False
        
        # Edge without required fields
        assert validate_graph({
            "nodes": [{"id": "1"}],
            "edges": [{"source": "1"}]  # Missing target and predicate
        }) is False
    
    def test_validate_graph_empty_valid(self):
        # Empty but valid structure
        graph = {"nodes": [], "edges": []}
        assert validate_graph(graph) is True


if __name__ == "__main__":
    pytest.main([__file__])
