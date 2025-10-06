"""Tests for two-stage node matching and edge mapping."""

import pytest
import numpy as np
from frame2kg_eval.matching.assign import two_stage_node_match, compute_edge_mapping
from frame2kg_eval.matching.text import TextSimilarityComputer


class TestTwoStageMatching:
    
    def test_perfect_matching(self):
        """Test when all nodes match perfectly."""
        p_nodes = [
            {"id": "p1", "label": "person", "location": [0.1, 0.1, 0.2, 0.2]},
            {"id": "b1", "label": "ball", "location": [0.5, 0.5, 0.6, 0.6]}
        ]
        g_nodes = [
            {"id": "g1", "label": "person", "location": [0.1, 0.1, 0.2, 0.2]},
            {"id": "g2", "label": "ball", "location": [0.5, 0.5, 0.6, 0.6]}
        ]
        
        result = two_stage_node_match(
            p_nodes, g_nodes,
            tau=0.3, alpha=0.7,
            text_mode="tfidf",
            text_fields=("id", "label"),
            text_floor=0.1
        )
        
        assert len(result["mapping"]) == 2
        assert 0 in result["mapping"]
        assert 1 in result["mapping"]
        assert len(result["unmatched_pred"]) == 0
        assert len(result["unmatched_gt"]) == 0
    
    def test_iou_gating(self):
        """Test that low IoU pairs are filtered out."""
        p_nodes = [
            {"id": "p1", "label": "person", "location": [0.0, 0.0, 0.1, 0.1]},
            {"id": "p2", "label": "person", "location": [0.9, 0.9, 1.0, 1.0]}
        ]
        g_nodes = [
            {"id": "g1", "label": "person", "location": [0.0, 0.0, 0.1, 0.1]},
            {"id": "g2", "label": "person", "location": [0.5, 0.5, 0.6, 0.6]}
        ]
        
        # High threshold should only match the first pair
        result = two_stage_node_match(
            p_nodes, g_nodes,
            tau=0.8,  # High IoU threshold
            alpha=0.5,
            text_mode="tfidf",
            text_fields=("label",),
            text_floor=0.1
        )
        
        assert len(result["mapping"]) == 1
        assert result["mapping"][0] == 0  # Only first nodes match
        assert 1 in result["unmatched_pred"]
        assert 1 in result["unmatched_gt"]
    
    def test_text_similarity_influence(self):
        """Test that text similarity affects matching."""
        p_nodes = [
            {"id": "car1", "label": "vehicle", "location": [0.1, 0.1, 0.3, 0.3]},
            {"id": "person1", "label": "human", "location": [0.11, 0.11, 0.31, 0.31]}
        ]
        g_nodes = [
            {"id": "auto1", "label": "vehicle", "location": [0.1, 0.1, 0.3, 0.3]},
            {"id": "man1", "label": "human", "location": [0.1, 0.1, 0.3, 0.3]}
        ]
        
        # Both predictions have similar IoU with ground truth
        # Text similarity should determine the matching
        result = two_stage_node_match(
            p_nodes, g_nodes,
            tau=0.5,
            alpha=0.3,  # Lower weight on IoU, higher on text
            text_mode="tfidf",
            text_fields=("label",),
            text_floor=0.1
        )
        
        # Should match based on labels (vehicle->vehicle, human->human)
        assert len(result["mapping"]) == 2
        # Check that labels match in the mapping
        for p_idx, g_idx in result["mapping"].items():
            assert p_nodes[p_idx]["label"] == g_nodes[g_idx]["label"]
    
    def test_empty_inputs(self):
        """Test handling of empty node lists."""
        # Empty predictions
        result = two_stage_node_match(
            [], [{"id": "g1", "label": "person"}],
            tau=0.3, alpha=0.7, text_mode="tfidf"
        )
        assert len(result["mapping"]) == 0
        assert len(result["unmatched_pred"]) == 0
        assert len(result["unmatched_gt"]) == 1
        
        # Empty ground truth
        result = two_stage_node_match(
            [{"id": "p1", "label": "person"}], [],
            tau=0.3, alpha=0.7, text_mode="tfidf"
        )
        assert len(result["mapping"]) == 0
        assert len(result["unmatched_pred"]) == 1
        assert len(result["unmatched_gt"]) == 0
        
        # Both empty
        result = two_stage_node_match(
            [], [],
            tau=0.3, alpha=0.7, text_mode="tfidf"
        )
        assert len(result["mapping"]) == 0
    
    def test_text_floor_filtering(self):
        """Test that text similarity floor filters out poor matches."""
        p_nodes = [
            {"id": "xyz123", "label": "qwerty", "location": [0.1, 0.1, 0.2, 0.2]}
        ]
        g_nodes = [
            {"id": "abc456", "label": "asdfgh", "location": [0.1, 0.1, 0.2, 0.2]}
        ]
        
        # Perfect IoU but completely different text
        result = two_stage_node_match(
            p_nodes, g_nodes,
            tau=0.5,
            alpha=0.5,
            text_mode="tfidf",
            text_fields=("id", "label"),
            text_floor=0.8  # High text similarity requirement
        )
        
        # Should not match due to low text similarity
        assert len(result["mapping"]) == 0
        assert 0 in result["unmatched_pred"]
        assert 0 in result["unmatched_gt"]
    
    def test_matrices_output(self):
        """Test that output matrices have correct shape and values."""
        p_nodes = [
            {"id": "p1", "label": "person", "location": [0.1, 0.1, 0.2, 0.2]},
            {"id": "p2", "label": "car", "location": [0.5, 0.5, 0.6, 0.6]}
        ]
        g_nodes = [
            {"id": "g1", "label": "person", "location": [0.1, 0.1, 0.2, 0.2]},
            {"id": "g2", "label": "vehicle", "location": [0.5, 0.5, 0.6, 0.6]}
        ]
        
        result = two_stage_node_match(
            p_nodes, g_nodes,
            tau=0.3, alpha=0.7,
            text_mode="tfidf",
            text_fields=("label",),
            text_floor=0.1
        )
        
        # Check matrices
        assert "matrices" in result
        assert "iou" in result["matrices"]
        assert "text_sim" in result["matrices"]
        assert "score" in result["matrices"]
        
        # Check shapes
        assert result["matrices"]["iou"].shape == (2, 2)
        assert result["matrices"]["text_sim"].shape == (2, 2)
        assert result["matrices"]["score"].shape == (2, 2)
        
        # Check IoU values
        assert result["matrices"]["iou"][0, 0] == 1.0  # Perfect overlap
        assert result["matrices"]["iou"][1, 1] == 1.0  # Perfect overlap
        assert result["matrices"]["iou"][0, 1] == 0.0  # No overlap
        assert result["matrices"]["iou"][1, 0] == 0.0  # No overlap


class TestEdgeMapping:
    
    def test_edge_mapping_exact(self):
        """Test exact predicate matching for edges."""
        p_edges = [
            {"source": "p1", "target": "p2", "predicate": "next_to"},
            {"source": "p2", "target": "p3", "predicate": "holding"},
            {"source": "p1", "target": "p3", "predicate": "near"}
        ]
        g_edges = [
            {"source": "g1", "target": "g2", "predicate": "next_to"},
            {"source": "g2", "target": "g3", "predicate": "holding"},
            {"source": "g1", "target": "g3", "predicate": "far"}  # Different predicate
        ]
        
        node_mapping = {"p1": "g1", "p2": "g2", "p3": "g3"}
        
        edge_mapping = compute_edge_mapping(p_edges, g_edges, node_mapping, "exact")
        
        assert len(edge_mapping) == 2  # Only first two edges match
        assert 0 in edge_mapping
        assert 1 in edge_mapping
        assert 2 not in edge_mapping  # "near" != "far"
    
    def test_edge_mapping_missing_nodes(self):
        """Test edge mapping when node mapping is incomplete."""
        p_edges = [
            {"source": "p1", "target": "p2", "predicate": "rel1"},
            {"source": "p2", "target": "p3", "predicate": "rel2"},
            {"source": "p3", "target": "p4", "predicate": "rel3"}
        ]
        g_edges = [
            {"source": "g1", "target": "g2", "predicate": "rel1"},
            {"source": "g2", "target": "g3", "predicate": "rel2"},
            {"source": "g3", "target": "g4", "predicate": "rel3"}
        ]
        
        # Incomplete mapping - p4 is not mapped
        node_mapping = {"p1": "g1", "p2": "g2", "p3": "g3"}
        
        edge_mapping = compute_edge_mapping(p_edges, g_edges, node_mapping, "exact")
        
        assert len(edge_mapping) == 2  # Only first two edges can be mapped
        assert 0 in edge_mapping
        assert 1 in edge_mapping
        assert 2 not in edge_mapping  # p4 is not mapped
    
    def test_edge_mapping_normalized_predicates(self):
        """Test semantic predicate matching with normalization."""
        p_edges = [
            {"source": "p1", "target": "p2", "predicate": "NEXT_TO"},
            {"source": "p2", "target": "p3", "predicate": "is-holding"}
        ]
        g_edges = [
            {"source": "g1", "target": "g2", "predicate": "next-to"},
            {"source": "g2", "target": "g3", "predicate": "is_holding"}
        ]
        
        node_mapping = {"p1": "g1", "p2": "g2", "p3": "g3"}
        
        # Semantic mode should normalize predicates
        edge_mapping = compute_edge_mapping(p_edges, g_edges, node_mapping, "semantic")
        
        assert len(edge_mapping) == 2  # Both should match after normalization
    
    def test_edge_mapping_empty(self):
        """Test edge mapping with empty inputs."""
        node_mapping = {"p1": "g1"}
        
        # Empty predictions
        edge_mapping = compute_edge_mapping([], [{"source": "g1", "target": "g1", "predicate": "self"}], node_mapping, "exact")
        assert len(edge_mapping) == 0
        
        # Empty ground truth
        edge_mapping = compute_edge_mapping([{"source": "p1", "target": "p1", "predicate": "self"}], [], node_mapping, "exact")
        assert len(edge_mapping) == 0


class TestTextSimilarity:
    
    def test_text_similarity_modes(self):
        """Test different text similarity computation modes."""
        computer_tfidf = TextSimilarityComputer(mode="tfidf")
        computer_semantic = TextSimilarityComputer(mode="semantic")
        
        nodes1 = [
            {"id": "person1", "label": "walking person"},
            {"id": "car1", "label": "red vehicle"}
        ]
        nodes2 = [
            {"id": "human1", "label": "person walking"},
            {"id": "auto1", "label": "vehicle red"}
        ]
        
        # TF-IDF similarity
        sim_tfidf = computer_tfidf.compute_similarity_matrix(nodes1, nodes2)
        assert sim_tfidf.shape == (2, 2)
        assert sim_tfidf[0, 0] > sim_tfidf[0, 1]  # person matches person better
        assert sim_tfidf[1, 1] > sim_tfidf[1, 0]  # vehicle matches vehicle better
        
        # Semantic similarity (if model available)
        try:
            sim_semantic = computer_semantic.compute_similarity_matrix(nodes1, nodes2)
            assert sim_semantic.shape == (2, 2)
            # Semantic should also match similar concepts
            assert sim_semantic[0, 0] > 0.5  # "walking person" similar to "person walking"
        except ImportError:
            # Skip if sentence-transformers not installed
            pass
    
    def test_extract_node_text(self):
        """Test text extraction from nodes."""
        computer = TextSimilarityComputer()
        
        node = {
            "id": "person123",
            "label": "standing_person",
            "attributes": {
                "appearance": "blue shirt",
                "size": "large"
            }
        }
        
        # Extract with default fields
        text = computer.extract_node_text(node, ("id", "label"))
        assert "person" in text  # ID without digits
        assert "standing" in text
        assert "blue shirt" in text  # Attributes included
        assert "large" in text
        
        # Extract with only label
        text_label = computer.extract_node_text(node, ("label",))
        assert "standing" in text_label
        assert "123" not in text_label  # No ID
    
    def test_cache_usage(self):
        """Test that embeddings are cached for efficiency."""
        computer = TextSimilarityComputer(mode="semantic")
        
        nodes1 = [{"id": "p1", "label": "test"}]
        nodes2 = [{"id": "g1", "label": "test"}]
        
        # First computation
        try:
            _ = computer.compute_similarity_matrix(nodes1, nodes2)
            cache_size_1 = len(computer._embedding_cache)
            
            # Second computation with same ground truth
            _ = computer.compute_similarity_matrix(nodes1, nodes2)
            cache_size_2 = len(computer._embedding_cache)
            
            # Cache should not grow (reused embeddings)
            assert cache_size_2 == cache_size_1
            
            # Clear cache
            computer.clear_cache()
            assert len(computer._embedding_cache) == 0
        except ImportError:
            # Skip if sentence-transformers not installed
            pass


if __name__ == "__main__":
    pytest.main([__file__])
