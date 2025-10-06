"""Tests for metrics computation."""

import pytest
from frame2kg_eval.metrics.nodes import node_prf1, aggregate_micro, aggregate_macro
from frame2kg_eval.metrics.edges import edge_prf1, edge_by_label_baseline


class TestNodeMetrics:
    
    def test_node_prf1_perfect_match(self):
        p_nodes = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        g_nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        mapping = {0: 0, 1: 1, 2: 2}  # All matched
        
        metrics = node_prf1(p_nodes, g_nodes, mapping)
        
        assert metrics["tp"] == 3
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
    
    def test_node_prf1_partial_match(self):
        p_nodes = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        g_nodes = [{"id": "a"}, {"id": "b"}]
        mapping = {0: 0}  # Only first matched
        
        metrics = node_prf1(p_nodes, g_nodes, mapping)
        
        assert metrics["tp"] == 1
        assert metrics["fp"] == 2  # 2 unmatched predictions
        assert metrics["fn"] == 1  # 1 unmatched ground truth
        assert metrics["precision"] == 1/3
        assert metrics["recall"] == 1/2
        assert abs(metrics["f1"] - 0.4) < 0.01
    
    def test_node_prf1_empty_cases(self):
        # No predictions
        metrics = node_prf1([], [{"id": "1"}], {})
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
        
        # No ground truth
        metrics = node_prf1([{"id": "1"}], [], {})
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0  # Convention: no GT means recall is 0
        assert metrics["f1"] == 0.0
    
    def test_aggregate_micro(self):
        metrics_list = [
            {"tp": 2, "fp": 1, "fn": 1, "support": 3},
            {"tp": 3, "fp": 2, "fn": 0, "support": 3},
            {"tp": 1, "fp": 0, "fn": 2, "support": 3}
        ]
        
        result = aggregate_micro(metrics_list)
        
        assert result["tp"] == 6  # 2+3+1
        assert result["fp"] == 3  # 1+2+0
        assert result["fn"] == 3  # 1+0+2
        assert result["precision"] == 6/9  # 6/(6+3)
        assert result["recall"] == 6/9  # 6/(6+3)
        assert abs(result["f1"] - 2/3) < 0.01
    
    def test_aggregate_macro(self):
        metrics_list = [
            {"precision": 0.5, "recall": 0.6, "f1": 0.545, "tp": 3, "fp": 3, "fn": 2, "support": 5},
            {"precision": 0.8, "recall": 0.7, "f1": 0.747, "tp": 7, "fp": 2, "fn": 3, "support": 10},
        ]
        
        result = aggregate_macro(metrics_list)
        
        assert abs(result["precision"] - 0.65) < 0.0001  # (0.5+0.8)/2
        assert abs(result["recall"] - 0.65) < 0.0001  # (0.6+0.7)/2
        assert abs(result["f1"] - 0.646) < 0.01  # (0.545+0.747)/2


class TestEdgeMetrics:
    
    def test_edge_prf1_with_mapping(self):
        p_edges = [
            {"source": "p1", "target": "p2", "predicate": "next_to"},
            {"source": "p2", "target": "p3", "predicate": "holding"}
        ]
        g_edges = [
            {"source": "g1", "target": "g2", "predicate": "next_to"},
            {"source": "g2", "target": "g3", "predicate": "holding"}
        ]
        node_mapping = {"p1": "g1", "p2": "g2", "p3": "g3"}
        
        metrics = edge_prf1(p_edges, g_edges, node_mapping, "exact")
        
        assert metrics["tp"] == 2
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
    
    def test_edge_prf1_partial_mapping(self):
        p_edges = [
            {"source": "p1", "target": "p2", "predicate": "next_to"},
            {"source": "p2", "target": "p3", "predicate": "holding"}
        ]
        g_edges = [
            {"source": "g1", "target": "g2", "predicate": "next_to"},
            {"source": "g2", "target": "g3", "predicate": "holding"}
        ]
        # Missing p3 mapping
        node_mapping = {"p1": "g1", "p2": "g2"}
        
        metrics = edge_prf1(p_edges, g_edges, node_mapping, "exact")
        
        assert metrics["tp"] == 1  # Only first edge matches
        assert metrics["fp"] == 1  # Second edge can't be mapped
        assert metrics["fn"] == 1  # Second GT edge not matched
    
    def test_edge_by_label_baseline(self):
        p_nodes = [
            {"id": "p1", "label": "person"},
            {"id": "p2", "label": "ball"}
        ]
        g_nodes = [
            {"id": "g1", "label": "person"},
            {"id": "g2", "label": "ball"}
        ]
        p_edges = [
            {"source": "p1", "target": "p2", "predicate": "holding"}
        ]
        g_edges = [
            {"source": "g1", "target": "g2", "predicate": "holding"}
        ]
        
        metrics = edge_by_label_baseline(p_edges, g_edges, p_nodes, g_nodes)
        
        assert metrics["tp"] == 1
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
