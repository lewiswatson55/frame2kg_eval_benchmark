"""Integration tests for I/O and CLI components."""

import pytest
import json
import tempfile
from pathlib import Path

from frame2kg_eval.cli.evaluate import load_config
from frame2kg_eval.io.preds import PredictionLoader, index_predictions
from frame2kg_eval.io.groundtruth import LocalJsonAdapter, create_ground_truth_adapter
from frame2kg_eval.io.schema import validate_graph, standardize_graph
from frame2kg_eval.metrics.validity import check_file_validity, compute_validity_from_directory
from frame2kg_eval.metrics.timing import manifest_timing, extract_timing_per_frame
from frame2kg_eval.utils.ids import parse_filename, build_filename, resolve_prediction_path


class TestFileIO:
    
    def test_prediction_loader(self):
        """Test loading predictions from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create valid prediction files
            valid_pred = {
                "nodes": [
                    {"id": "p1", "label": "person", "location": "0.1,0.2,0.3,0.4"}
                ],
                "edges": [
                    {"source": "p1", "target": "p1", "predicate": "self"}
                ]
            }
            
            # Write valid JSON file
            json_path = tmppath / "video1.001.json"
            with open(json_path, 'w') as f:
                json.dump(valid_pred, f)
            
            # Write invalid JSON file
            invalid_path = tmppath / "video1.002.json"
            with open(invalid_path, 'w') as f:
                f.write("not valid json {")
            
            # Write raw text file
            raw_path = tmppath / "video1.003.raw.txt"
            with open(raw_path, 'w') as f:
                f.write("raw output")
            
            # Load predictions
            loader = PredictionLoader(tmppath)
            
            # Check index
            index = loader.get_index()
            assert len(index) == 3
            assert ("video1", 1) in index
            assert ("video1", 2) in index
            assert ("video1", 3) in index
            
            # Get valid graph
            graph = loader.get_graph("video1", 1)
            assert graph is not None
            assert len(graph["nodes"]) == 1
            assert len(graph["edges"]) == 1
            
            # Get invalid graph
            graph_invalid = loader.get_graph("video1", 2)
            assert graph_invalid is None
            
            # Get raw text (should return None)
            graph_raw = loader.get_graph("video1", 3)
            assert graph_raw is None
            
            # Count validity
            valid, invalid, total = loader.count_valid()
            assert valid == 1
            assert invalid == 2
            assert total == 3
    
    def test_ground_truth_adapter(self):
        """Test local JSON ground truth adapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create ground truth files
            gt_data = {
                "nodes": [
                    {"id": "g1", "label": "person"},
                    {"id": "g2", "label": "ball"}
                ],
                "edges": [
                    {"source": "g1", "target": "g2", "predicate": "holding"}
                ]
            }
            
            # Write GT files
            for i in range(3):
                gt_path = tmppath / f"video1.{i:03d}.json"
                with open(gt_path, 'w') as f:
                    json.dump(gt_data, f)
            
            # Create adapter
            adapter = LocalJsonAdapter(tmppath)
            
            # Check frame count
            assert adapter.count_frames() == 3
            
            # Get specific frame
            graph = adapter.get_graph("video1", 0)
            assert graph is not None
            assert len(graph["nodes"]) == 2
            assert len(graph["edges"]) == 1
            
            # Iterate frames
            frames = list(adapter.iter_frames())
            assert len(frames) == 3
            for vid, fno, graph in frames:
                assert vid == "video1"
                assert 0 <= fno <= 2
                assert len(graph["nodes"]) == 2
    
    def test_ground_truth_factory(self):
        """Test ground truth adapter factory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create a test file
            test_file = tmppath / "test.001.json"
            with open(test_file, 'w') as f:
                json.dump({"nodes": [], "edges": []}, f)
            
            # Test local path adapter
            adapter = create_ground_truth_adapter(str(tmppath))
            assert isinstance(adapter, LocalJsonAdapter)
            assert adapter.count_frames() == 1
            
            # Test HuggingFace spec parsing
            # Note: Don't actually load dataset to avoid network dependency
            with pytest.raises(Exception):
                # This will fail without network/dataset access
                adapter_hf = create_ground_truth_adapter("hf:fake_dataset:split")
    
    def test_id_parsing(self):
        """Test filename parsing utilities."""
        # Valid filenames
        assert parse_filename("video1.001.json") == ("video1", 1)
        assert parse_filename("complex_name.0042.json") == ("complex_name", 42)
        assert parse_filename("test.123.raw.txt") == ("test", 123)
        
        # With path
        assert parse_filename(Path("/path/to/video.001.json")) == ("video", 1)
        
        # Invalid filenames
        assert parse_filename("invalid.json") is None
        assert parse_filename("no_extension") is None
        assert parse_filename("missing.frame.json") is None
        
        # Build filename
        assert build_filename("video1", 42, is_json=True) == "video1.42.json"
        assert build_filename("video1", 42, is_json=False) == "video1.42.raw.txt"
    
    def test_resolve_prediction_path(self):
        """Test prediction path resolution with fallbacks."""
        pred_index = {
            ("video1", 1): Path("video1.001.json"),
            ("video1", 2): Path("video1.002.json"),
            ("video2", 10): Path("video2.010.json")
        }
        
        # Exact match
        assert resolve_prediction_path("video1", 1, pred_index) == Path("video1.001.json")
        
        # Off-by-one fallback
        assert resolve_prediction_path("video1", 3, pred_index, allow_off_by_one=True) == Path("video1.002.json")
        assert resolve_prediction_path("video2", 11, pred_index, allow_off_by_one=True) == Path("video2.010.json")
        
        # No fallback
        assert resolve_prediction_path("video1", 3, pred_index, allow_off_by_one=False) is None
        
        # Missing video
        assert resolve_prediction_path("video3", 1, pred_index) is None
    
    def test_validity_checking(self):
        """Test JSON validity checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Valid JSON
            valid_file = tmppath / "valid.001.json"
            with open(valid_file, 'w') as f:
                json.dump({
                    "nodes": [{"id": "n1"}],
                    "edges": []
                }, f)
            
            # Invalid JSON
            invalid_file = tmppath / "invalid.001.json"
            with open(invalid_file, 'w') as f:
                f.write("{invalid json")
            
            # Raw text file
            raw_file = tmppath / "raw.001.raw.txt"
            with open(raw_file, 'w') as f:
                f.write("raw text")
            
            # Check individual files
            assert check_file_validity(valid_file) is True
            assert check_file_validity(invalid_file) is False
            assert check_file_validity(raw_file) is False
            
            # Check directory
            stats = compute_validity_from_directory(tmppath)
            assert stats["valid_count"] == 1
            assert stats["invalid_count"] == 2  # invalid JSON + raw text
            assert stats["total_count"] == 3
            assert 30 < stats["validity_rate"] < 40  # ~33.3%
    
    def test_manifest_timing(self):
        """Test timing extraction from manifest files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create manifest CSV
            manifest_path = tmppath / "manifest.csv"
            with open(manifest_path, 'w') as f:
                f.write("video_id,frame_number,gen_wall_time_s\n")
                f.write("video1,1,1.5\n")
                f.write("video1,2,2.0\n")
                f.write("video1,3,1.0\n")
                f.write("video2,1,3.0\n")
            
            # Extract timing statistics
            stats = manifest_timing(manifest_path)
            assert stats["n"] == 4
            assert stats["mean"] == 1.875  # (1.5 + 2.0 + 1.0 + 3.0) / 4
            assert stats["min"] == 1.0
            assert stats["max"] == 3.0
            assert stats["median"] == 1.75  # Between 1.5 and 2.0
            
            # Extract per-frame timing
            frame_times = extract_timing_per_frame(manifest_path)
            assert len(frame_times) == 4
            assert frame_times[("video1", 1)] == 1.5
            assert frame_times[("video1", 2)] == 2.0
            assert frame_times[("video2", 1)] == 3.0
            
            # Non-existent manifest
            fake_path = tmppath / "fake.csv"
            stats_empty = manifest_timing(fake_path)
            assert stats_empty["n"] == 0
            assert stats_empty["mean"] is None


class TestCLIImports:
    """Test that CLI modules can be imported."""
    
    def test_import_evaluate(self):
        from frame2kg_eval.cli import evaluate
        assert hasattr(evaluate, 'main')
    
    def test_import_sweep(self):
        from frame2kg_eval.cli import sweep
        assert hasattr(sweep, 'main')
    
    def test_import_aggregate(self):
        from frame2kg_eval.cli import aggregate
        assert hasattr(aggregate, 'main')
    
    def test_import_doctor(self):
        from frame2kg_eval.cli import doctor
        assert hasattr(doctor, 'main')

    def test_default_config_uses_label_text_field(self):
        config = load_config()
        assert config["text_fields"] == ["label", "attributes"]


class TestSchemaValidation:
    """Test graph schema validation and standardization."""
    
    def test_validate_various_graphs(self):
        """Test validation of different graph structures."""
        # Valid minimal graph
        assert validate_graph({"nodes": [], "edges": []})
        
        # Valid complete graph
        assert validate_graph({
            "nodes": [
                {"id": "n1", "label": "person"},
                {"id": "n2", "label": "ball"}
            ],
            "edges": [
                {"source": "n1", "target": "n2", "predicate": "holding"}
            ]
        })
        
        # Valid with only nodes (no edges)
        assert validate_graph({"nodes": [{"id": "n1"}]})
        
        # Valid with only edges (no nodes) 
        assert validate_graph({"edges": [{"source": "n1", "target": "n2", "predicate": "rel"}]})
        
        # Invalid: completely empty
        assert not validate_graph({})  # Neither nodes nor edges
        
        # Invalid node structure
        assert not validate_graph({
            "nodes": [{"label": "no_id"}],  # Missing id
            "edges": []
        })
        
        # Invalid edge structure
        assert not validate_graph({
            "nodes": [{"id": "n1"}],
            "edges": [{"source": "n1"}]  # Missing target and predicate
        })
    
    def test_standardize_various_inputs(self):
        """Test standardization of different input formats."""
        # Mixed location formats
        raw_graph = {
            "nodes": [
                {"id": 123, "label": "PERSON", "location": "0.1,0.2,0.3,0.4,0.95"},
                {"id": "ball_1", "label": None, "location": [0.5, 0.6, 0.7, 0.8]},
                {"id": "car", "label": "vehicle"}  # No location
            ],
            "edges": [
                {"source": 123, "target": "ball_1", "predicate": "NEAR"}
            ]
        }
        
        std_graph = standardize_graph(raw_graph)
        
        # Check nodes
        assert len(std_graph["nodes"]) == 3
        assert std_graph["nodes"][0]["id"] == "123"
        assert std_graph["nodes"][0]["label"] == "PERSON"
        assert std_graph["nodes"][0]["location"] == [0.1, 0.2, 0.3, 0.4]
        assert std_graph["nodes"][0]["conf"] == 0.95
        
        assert std_graph["nodes"][1]["id"] == "ball_1"
        assert std_graph["nodes"][1]["label"] == "None"
        assert std_graph["nodes"][1]["location"] == [0.5, 0.6, 0.7, 0.8]
        assert std_graph["nodes"][1]["conf"] is None
        
        assert std_graph["nodes"][2]["location"] is None
        
        # Check edges
        assert len(std_graph["edges"]) == 1
        assert std_graph["edges"][0]["source"] == "123"
        assert std_graph["edges"][0]["target"] == "ball_1"
        assert std_graph["edges"][0]["predicate"] == "NEAR"


if __name__ == "__main__":
    pytest.main([__file__])
