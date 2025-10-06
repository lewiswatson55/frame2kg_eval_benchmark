"""Tests for normalization utilities."""

import pytest
from frame2kg_eval.utils.normalise import (
    normalise_label, normalise_id, parse_location, 
    extract_confidence, normalise_predicate
)


class TestNormalisation:
    
    def test_normalise_label(self):
        # Basic normalization
        assert normalise_label("Person_Walking") == "person walking"
        assert normalise_label("UPPER-CASE") == "upper case"
        assert normalise_label("with@special#chars") == "with special chars"
        assert normalise_label("  multiple   spaces  ") == "multiple spaces"
        
        # Edge cases
        assert normalise_label(None) == ""
        assert normalise_label("") == ""
        assert normalise_label(123) == "123"
        assert normalise_label("CamelCaseText") == "camelcasetext"
        assert normalise_label("snake_case_text") == "snake case text"
        assert normalise_label("!!!???") == ""  # All special chars removed
    
    def test_normalise_id(self):
        # ID normalization removes digits
        assert normalise_id("person123") == "person"
        assert normalise_id("object_42") == "object"
        assert normalise_id("item-99-test") == "item test"
        assert normalise_id("123pure_numbers456") == "pure numbers"
        
        # Edge cases
        assert normalise_id(None) == ""
        assert normalise_id("") == ""
        assert normalise_id("12345") == ""  # All digits removed
        assert normalise_id("node_1_person_2") == "node person"
    
    def test_normalise_predicate(self):
        # Predicates use same normalization as labels
        assert normalise_predicate("next_to") == "next to"
        assert normalise_predicate("HOLDING") == "holding"
        assert normalise_predicate("is-part-of") == "is part of"
    
    def test_parse_location(self):
        # Valid standard locations
        assert parse_location("0.1,0.2,0.3,0.4") == (0.1, 0.2, 0.3, 0.4)
        assert parse_location("0.1,0.2,0.3,0.4,0.95") == (0.1, 0.2, 0.3, 0.4)  # With confidence
        assert parse_location([0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4)
        assert parse_location((0.1, 0.2, 0.3, 0.4)) == (0.1, 0.2, 0.3, 0.4)  # Tuple input
        
        # Clamping to [0, 1] range
        assert parse_location("-0.1,0.2,0.3,0.4") == (0.0, 0.2, 0.3, 0.4)  # Negative clamped to 0
        assert parse_location("0.8,0.9,1.1,1.2") == (0.8, 0.9, 1.0, 1.0)  # > 1 clamped to 1
        
        # Order correction (swap if x1>x2 or y1>y2)
        assert parse_location("0.3,0.4,0.1,0.2") == (0.1, 0.2, 0.3, 0.4)  # Both swapped
        assert parse_location("0.3,0.2,0.1,0.4") == (0.1, 0.2, 0.3, 0.4)  # Just x swapped
        assert parse_location("0.1,0.4,0.3,0.2") == (0.1, 0.2, 0.3, 0.4)  # Just y swapped
        
        # Complex case: swap then clamp
        # 1.5 > 0.3 so swap x: (0.3, 0.2, 1.5, 0.4) then clamp: (0.3, 0.2, 1.0, 0.4)
        assert parse_location("1.5,0.2,0.3,0.4") == (0.3, 0.2, 1.0, 0.4)
        
        # Zero-area box handling (adds small epsilon)
        result = parse_location("0.5,0.5,0.5,0.5")  # Point becomes small box
        assert result[0] == 0.5
        assert result[1] == 0.5
        assert result[2] > 0.5  # x2 = x1 + epsilon
        assert result[3] > 0.5  # y2 = y1 + epsilon
        
        # Invalid locations
        assert parse_location(None) is None
        assert parse_location("") is None
        assert parse_location("invalid") is None
        assert parse_location("0.1,0.2") is None  # Too few values
        assert parse_location("0.1") is None
        assert parse_location([0.1, 0.2]) is None  # Too few values in list
        assert parse_location({}) is None  # Wrong type
    
    def test_extract_confidence(self):
        # Extract 5th value as confidence
        assert extract_confidence("0.1,0.2,0.3,0.4,0.95") == 0.95
        assert extract_confidence([0.1, 0.2, 0.3, 0.4, 0.8]) == 0.8
        assert extract_confidence((0.1, 0.2, 0.3, 0.4, 0.99)) == 0.99
        
        # No confidence value
        assert extract_confidence("0.1,0.2,0.3,0.4") is None
        assert extract_confidence([0.1, 0.2, 0.3, 0.4]) is None
        
        # Invalid input
        assert extract_confidence(None) is None
        assert extract_confidence("") is None
        assert extract_confidence("invalid") is None


if __name__ == "__main__":
    pytest.main([__file__])
