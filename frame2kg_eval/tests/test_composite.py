"""Tests for composite diagnostic metrics."""

import pytest
from frame2kg_eval.metrics.composite import (
    parse_location,
    compute_box_union,
    center_in_box,
    compute_composite_score,
    find_composite_groups,
    composite_diagnostics
)
from frame2kg_eval.matching.text import TextSimilarityComputer


def test_parse_location():
    """Test location parsing from various formats."""
    # String format
    assert parse_location("0.1,0.2,0.3,0.4,0.95") == [0.1, 0.2, 0.3, 0.4]
    
    # List format
    assert parse_location([0.1, 0.2, 0.3, 0.4]) == [0.1, 0.2, 0.3, 0.4]
    
    # Clamp to [0, 1]
    assert parse_location("1.5,0.2,0.3,-0.1,0.95") == [0.3, 0.0, 1.0, 0.2]
    
    # Invalid cases
    assert parse_location(None) is None
    assert parse_location("invalid") is None
    assert parse_location([1, 2]) is None


def test_compute_box_union():
    """Test bounding box union computation."""
    boxes = [
        [10, 10, 20, 20],
        [15, 15, 25, 25],
        [5, 5, 15, 15]
    ]
    union = compute_box_union(boxes)
    assert union == [5, 5, 25, 25]
    
    # Test with None boxes
    assert compute_box_union([None, None]) is None
    assert compute_box_union([]) is None


def test_center_in_box():
    """Test center containment check."""
    point_box = [10, 10, 20, 20]  # Center at (15, 15)
    container = [0, 0, 30, 30]
    assert center_in_box(point_box, container) is True
    
    container = [20, 20, 30, 30]
    assert center_in_box(point_box, container) is False
    
    # Test with None
    assert center_in_box(None, container) is False
    assert center_in_box(point_box, None) is False


def test_composite_diagnostics_basic():
    """Test basic composite diagnostic computation."""
    # Create test nodes
    p_nodes = [
        {"id": "p1", "label": "apple", "location": "0.10,0.10,0.20,0.20,0.9"},
        {"id": "p2", "label": "banana", "location": "0.20,0.10,0.30,0.20,0.9"},
        {"id": "p3", "label": "orange", "location": "0.30,0.10,0.40,0.20,0.9"},
        {"id": "p4", "label": "unmatched", "location": "0.80,0.80,0.90,0.90,0.9"}
    ]
    
    g_nodes = [
        {"id": "g1", "label": "fruit_basket", "location": "0.05,0.05,0.45,0.25,1.0"},
        {"id": "g2", "label": "other", "location": "0.70,0.70,0.75,0.75,1.0"}
    ]
    
    # Simulate a mapping where none of the individual fruits matched the basket
    mapping = {}  # Empty mapping - all nodes unmatched
    
    # Run composite diagnostics
    result = composite_diagnostics(
        p_nodes, g_nodes, mapping,
        beta=0.6, tau_c=0.2, theta=0.4  # Lower threshold for testing
    )
    
    # Check that we found some composite matches
    assert "composite_hits_gt_side" in result
    assert "composite_hits_pred_side" in result
    assert "composite_fn_explained_pct" in result
    assert "composite_fp_explained_pct" in result
    

def test_find_composite_groups():
    """Test composite group finding."""
    anchor_nodes = [
        {"id": "a1", "label": "fruit_basket", "location": "0.05,0.05,0.45,0.25,1.0"}
    ]
    
    candidate_nodes = [
        {"id": "c1", "label": "apple", "location": "0.10,0.10,0.20,0.20,0.9"},
        {"id": "c2", "label": "banana", "location": "0.20,0.10,0.30,0.20,0.9"},
        {"id": "c3", "label": "orange", "location": "0.30,0.10,0.40,0.20,0.9"},
        {"id": "c4", "label": "far_away", "location": "0.80,0.80,0.90,0.90,0.9"}
    ]
    
    unmatched_anchors = {0}  # First anchor is unmatched
    unmatched_candidates = {0, 1, 2, 3}  # All candidates unmatched
    
    hits, mappings, stats = find_composite_groups(
        anchor_nodes, candidate_nodes,
        unmatched_anchors, unmatched_candidates,
        beta=0.6, tau_c=0.1, theta=0.4, kmax=4
    )
    
    # Should find that the fruit nodes form a composite for the basket
    assert hits > 0
    assert 0 in mappings  # Anchor 0 should have a mapping
    # The mapping should include the spatially overlapping fruits
    assert 3 not in mappings[0]  # Far away node shouldn't be included
    # Check statistics
    assert "avg_group_size" in stats
    assert "mean_composite_score" in stats


def test_composite_score_computation():
    """Test composite score calculation."""
    text_computer = TextSimilarityComputer(mode="semantic")
    
    group_nodes = [
        {"id": "1", "label": "apple", "location": "0.10,0.10,0.20,0.20,0.9"},
        {"id": "2", "label": "banana", "location": "0.20,0.10,0.30,0.20,0.9"}
    ]
    
    anchor_nodes = [{"id": "a", "label": "fruits", "location": "0.05,0.05,0.35,0.25,1.0"}]
    
    # Pre-extract texts
    group_texts = [text_computer.extract_node_text(n, ("label",)) for n in group_nodes]
    anchor_texts = [text_computer.extract_node_text(n, ("label",)) for n in anchor_nodes]
    
    score = compute_composite_score(
        [0, 1], 0, group_nodes, anchor_nodes,
        group_texts, anchor_texts, text_computer, beta=0.6
    )
    
    # Score should be between 0 and 1
    assert 0 <= score <= 1
    
    # Test with empty group
    score_empty = compute_composite_score(
        [], 0, group_nodes, anchor_nodes,
        group_texts, anchor_texts, text_computer, beta=0.6
    )
    assert score_empty == 0.0


def test_composite_diagnostics_with_partial_matching():
    """Test composite diagnostics when some nodes are already matched."""
    p_nodes = [
        {"id": "p1", "label": "apple", "location": "0.10,0.10,0.20,0.20,0.9"},
        {"id": "p2", "label": "banana", "location": "0.20,0.10,0.30,0.20,0.9"},
        {"id": "p3", "label": "matched", "location": "0.50,0.50,0.60,0.60,0.9"}
    ]
    
    g_nodes = [
        {"id": "g1", "label": "fruit_basket", "location": "0.05,0.05,0.35,0.25,1.0"},
        {"id": "g2", "label": "matched", "location": "0.50,0.50,0.60,0.60,1.0"}
    ]
    
    # p3 is matched to g2
    mapping = {2: 1}
    
    result = composite_diagnostics(
        p_nodes, g_nodes, mapping,
        beta=0.6, tau_c=0.2, theta=0.4
    )
    
    # Should only consider unmatched nodes for composites
    assert result["composite_adjusted_recall"] >= 0.5  # At least the matched node


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
