"""Tests for handling missing or invalid prediction files."""

from frame2kg_eval.cli.aggregate import evaluate_single_run


def _make_gt_graphs():
    """Create a minimal ground-truth graph dictionary."""

    return {
        ("video1", 1): {
            "nodes": [
                {"id": "n1", "label": "person", "location": [0.1, 0.1, 0.2, 0.2]},
            ],
            "edges": [],
        }
    }


def _base_config(**overrides):
    config = {
        "tau": 0.3,
        "alpha": 0.7,
        "text_mode": "tfidf",
        "text_fields": ["id", "label"],
        "text_floor": 0.25,
        "predicate_mode": "exact",
        "include_invalid": True,
        "strict_mode": False,
    }
    config.update(overrides)
    return config


def test_missing_prediction_counts_as_empty(tmp_path):
    """A missing prediction file still produces a scored frame with zero F1."""

    pred_dir = tmp_path / "missing"
    pred_dir.mkdir()

    result = evaluate_single_run(pred_dir, _make_gt_graphs(), _base_config())

    assert result["num_frames"] == 1
    assert result["node_f1"] == 0.0
    assert result["edge_f1"] == 0.0


def test_invalid_json_counts_as_empty(tmp_path):
    """Invalid JSON predictions are treated as empty outputs."""

    pred_dir = tmp_path / "invalid"
    pred_dir.mkdir()
    (pred_dir / "video1.001.json").write_text("{not valid", encoding="utf-8")

    result = evaluate_single_run(pred_dir, _make_gt_graphs(), _base_config())

    assert result["invalid_count"] == 1
    assert result["node_f1"] == 0.0
    assert result["edge_f1"] == 0.0
