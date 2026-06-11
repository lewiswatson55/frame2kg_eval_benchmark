"""Compressed-token prediction loading and parsing."""

import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from frame2kg_eval.io.schema import standardize_graph, validate_graph
from frame2kg_eval.metrics.conformity import check_graph_schema, schema_conformity


GRAPH_TOKEN = "<|graph|>"
NODES_TOKEN = "<|nodes|>"
NODE_TOKEN = "<|node|>"
EDGES_TOKEN = "<|edges|>"
EDGE_TOKEN = "<|edge|>"
END_GRAPH_TOKEN = "<|end_graph|>"

_FILENAME_RE = re.compile(r"^(.+?)\.(\d+)\.graph\.txt$")
_FIELD_TOKEN_RE = re.compile(
    r"(<\|(?:id|label|bbox|conf|attrs|src|pred|tgt)\|>)"
)
_FIELD_NAMES = {
    "<|id|>": "id",
    "<|label|>": "label",
    "<|bbox|>": "bbox",
    "<|conf|>": "conf",
    "<|src|>": "src",
    "<|pred|>": "pred",
    "<|tgt|>": "tgt",
}


class CompressedTokenParseError(ValueError):
    """Raised when a compressed-token stream cannot be parsed as a graph."""


def parse_compressed_token_filename(filename: str) -> Optional[Tuple[str, int]]:
    """Parse ``<video_id>.<frame_no>.graph.txt`` filenames."""

    name = Path(filename).name
    match = _FILENAME_RE.match(name)
    if not match:
        return None

    return match.group(1), int(match.group(2))


def parse_compressed_token_graph(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse a compressed-token graph stream into the evaluator graph schema."""

    graph_start = text.find(GRAPH_TOKEN)
    if graph_start < 0:
        raise CompressedTokenParseError("missing graph token")

    graph_end = text.find(END_GRAPH_TOKEN, graph_start + len(GRAPH_TOKEN))
    if graph_end < 0:
        raise CompressedTokenParseError("missing end_graph token")

    body = text[graph_start + len(GRAPH_TOKEN):graph_end]
    nodes_start = body.find(NODES_TOKEN)
    edges_start = body.find(EDGES_TOKEN)

    if nodes_start < 0:
        raise CompressedTokenParseError("missing nodes token")
    if edges_start < 0:
        raise CompressedTokenParseError("missing edges token")
    if edges_start < nodes_start:
        raise CompressedTokenParseError("edges token precedes nodes token")

    nodes_section = body[nodes_start + len(NODES_TOKEN):edges_start]
    edges_section = body[edges_start + len(EDGES_TOKEN):]

    nodes = []
    for chunk in nodes_section.split(NODE_TOKEN)[1:]:
        if chunk.strip():
            nodes.append(_parse_node_chunk(chunk))

    edges = []
    for chunk in edges_section.split(EDGE_TOKEN)[1:]:
        if chunk.strip():
            edges.append(_parse_edge_chunk(chunk))

    return {"nodes": nodes, "edges": edges}


def load_compressed_token_graph(filepath: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Read and parse a compressed-token graph file."""

    with open(filepath, "r") as f:
        return parse_compressed_token_graph(f.read())


class CompressedTokenPredictionLoader:
    """Load and index compressed-token prediction files from a directory."""

    def __init__(self, pred_dir: Path):
        """Initialize loader with prediction directory."""

        self.pred_dir = Path(pred_dir)
        if not self.pred_dir.exists():
            raise ValueError(f"Prediction directory does not exist: {pred_dir}")

        self._index = {}
        self._build_index()

    def _build_index(self):
        """Build index of available compressed-token prediction files."""

        for filepath in self.pred_dir.glob("*.graph.txt"):
            parsed = parse_compressed_token_filename(filepath.name)
            if parsed:
                video_id, frame_no = parsed
                self._index[(video_id, frame_no)] = filepath

    def get_graph(self, video_id: str, frame_no: int) -> Optional[Dict]:
        """Get standardized graph for a specific frame."""

        key = (video_id, frame_no)
        if key not in self._index:
            return None

        filepath = self._index[key]

        try:
            raw_graph = load_compressed_token_graph(filepath)
            if not validate_graph(raw_graph):
                return None
            return standardize_graph(raw_graph)
        except (CompressedTokenParseError, IOError, UnicodeDecodeError):
            return None

    def iter_predictions(self) -> Iterator[Tuple[str, int, Optional[Dict], Path]]:
        """Iterate over all compressed-token predictions."""

        for (video_id, frame_no), filepath in sorted(self._index.items()):
            graph = self.get_graph(video_id, frame_no)
            yield video_id, frame_no, graph, filepath

    def get_index(self) -> Dict[Tuple[str, int], Path]:
        """Get the full file index."""

        return self._index.copy()

    def count_valid(self) -> Tuple[int, int, int]:
        """Count valid, invalid, and total compressed-token files."""

        valid = 0
        invalid = 0

        for vid, fno in self._index:
            graph = self.get_graph(vid, fno)
            if graph is not None:
                valid += 1
            else:
                invalid += 1

        return valid, invalid, valid + invalid


def check_compressed_token_file_validity(filepath: Path) -> bool:
    """Check if a compressed-token file contains a valid graph structure."""

    try:
        data = load_compressed_token_graph(filepath)
        return validate_graph(data)
    except (CompressedTokenParseError, IOError, UnicodeDecodeError):
        return False


def compute_compressed_token_validity_from_directory(pred_dir: Path) -> Dict:
    """Compute validity statistics for compressed-token prediction files."""

    file_records = []

    for filepath in pred_dir.glob("*.graph.txt"):
        parsed = parse_compressed_token_filename(filepath.name)
        if parsed:
            file_records.append({
                "path": filepath,
                "valid": check_compressed_token_file_validity(filepath),
                "video_id": parsed[0],
                "frame_no": parsed[1],
            })

    valid_count = sum(1 for record in file_records if record["valid"])
    invalid_count = len(file_records) - valid_count
    total_count = len(file_records)
    validity_rate = (valid_count / total_count * 100) if total_count > 0 else 0.0

    return {
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "total_count": total_count,
        "validity_rate": validity_rate,
    }


def check_compressed_token_file_conformity(
    filepath: Path,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check if a compressed-token file parses to a schema-conformant graph."""

    try:
        data = load_compressed_token_graph(filepath)
        return check_graph_schema(data)
    except (CompressedTokenParseError, IOError, UnicodeDecodeError):
        return False, None


def compute_compressed_token_conformity_from_directory(pred_dir: Path) -> Dict:
    """Compute schema conformity statistics for compressed-token predictions."""

    file_records = []
    conformity_issues = {}

    for filepath in pred_dir.glob("*.graph.txt"):
        parsed = parse_compressed_token_filename(filepath.name)
        if not parsed:
            continue

        valid_graph = check_compressed_token_file_validity(filepath)
        conformant, report = check_compressed_token_file_conformity(filepath)
        record = {
            "path": filepath,
            "valid_json": valid_graph,
            "conformant": conformant,
            "video_id": parsed[0],
            "frame_no": parsed[1],
        }
        file_records.append(record)

        if valid_graph and not conformant and report:
            key = f"{parsed[0]}.{parsed[1]}"
            conformity_issues[key] = report["issues"][:5]

    stats = schema_conformity(file_records)
    stats["sample_issues"] = conformity_issues

    return stats


def _parse_node_chunk(chunk: str) -> Dict[str, Any]:
    fields, raw_attrs = _parse_tagged_fields(chunk)

    node: Dict[str, Any] = {}
    if "id" in fields:
        node["id"] = fields["id"]
    if "label" in fields:
        node["label"] = fields["label"]

    bbox = _parse_bbox(fields.get("bbox"))
    conf = _parse_conf(fields.get("conf"))
    if bbox is not None:
        node["location"] = _format_location(bbox, conf)

    attrs = _parse_attrs(raw_attrs)
    if attrs:
        node["attributes"] = attrs

    return node


def _parse_edge_chunk(chunk: str) -> Dict[str, Any]:
    fields, _ = _parse_tagged_fields(chunk)

    edge: Dict[str, Any] = {}
    if "src" in fields:
        edge["source"] = fields["src"]
    if "tgt" in fields:
        edge["target"] = fields["tgt"]
    if "pred" in fields:
        edge["predicate"] = fields["pred"]

    return edge


def _parse_tagged_fields(chunk: str) -> Tuple[Dict[str, str], List[str]]:
    fields: Dict[str, str] = {}
    attrs: List[str] = []
    current_token = None

    for part in _FIELD_TOKEN_RE.split(chunk):
        if not part:
            continue

        if _FIELD_TOKEN_RE.fullmatch(part):
            current_token = part
            continue

        if current_token is None:
            continue

        value = part.strip()
        if current_token == "<|attrs|>":
            if value:
                attrs.append(value)
        elif value:
            fields[_FIELD_NAMES[current_token]] = value

        current_token = None

    return fields, attrs


def _parse_attrs(raw_attrs: List[str]) -> Dict[str, str]:
    attrs: Dict[str, str] = {}

    for index, raw_attr in enumerate(raw_attrs, start=1):
        if "=" in raw_attr:
            key, value = raw_attr.split("=", 1)
            key = key.strip()
            value = value.strip()
        else:
            key = f"attr_{index}"
            value = raw_attr.strip()

        if not key or not value:
            continue

        if key in attrs:
            attrs[key] = f"{attrs[key]}; {value}"
        else:
            attrs[key] = value

    return attrs


def _parse_bbox(value: Optional[str]) -> Optional[List[float]]:
    if value is None:
        return None

    parts = [part for part in re.split(r"[\s,]+", value.strip()) if part]
    if len(parts) < 4:
        return None

    try:
        return [float(part) for part in parts[:4]]
    except ValueError:
        return None


def _parse_conf(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value.strip())
    except ValueError:
        return None


def _format_location(bbox: List[float], conf: Optional[float]) -> str:
    values = list(bbox)
    if conf is not None:
        values.append(conf)
    return ",".join(format(value, "g") for value in values)
