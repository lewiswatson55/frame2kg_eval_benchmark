#!/usr/bin/env python3
"""Aggregate validation-dev evaluation metrics and select checkpoint winners."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from frame2kg_eval.metrics.conformity import compute_conformity_from_directory
from frame2kg_eval.metrics.validity import compute_validity_from_directory

# Mapping from human-friendly slug to directory name used under preds/
VARIANT_DIR_MAP: Dict[str, str] = {
    "qkvo": "QKVO",
    "qkvo_gate": "QKVO-Gate",
}

# Canonical checkpoint ordering for readability
CKPT_ORDER: Dict[str, int] = {
    "step1k": 0,
    "step2k": 1,
    "best": 2,
    "final": 3,
}


@dataclass
class EvalRecord:
    size: str
    variant: str
    ckpt: str
    result_path: Path
    pred_dir: Path
    node_micro_f1: float
    edge_micro_f1: float
    box_mean_iou: float
    box_iou_0p5_coverage: float
    box_iou_0p75_coverage: float
    validity_rate: float
    conformity_rate: float
    node_macro_f1: float
    extra: Dict[str, float] = field(default_factory=dict)
    winner: bool = False

    def ranking_key(self) -> Tuple[float, float, float, float, float, float]:
        """Return tuple used for tie-breaking in descending order."""
        return (
            safe_float(self.node_micro_f1),
            safe_float(self.edge_micro_f1),
            safe_float(self.box_mean_iou),
            safe_float(self.validity_rate),
            safe_float(self.conformity_rate),
            safe_float(self.node_macro_f1),
        )


def safe_float(value: float) -> float:
    """Treat NaN as -inf so higher-is-better ordering still works."""
    return value if not math.isnan(value) else float("-inf")


def parse_summary_rows(result_path: Path) -> Dict[str, Dict[str, float]]:
    """Extract micro/macro summary rows from a result CSV."""
    summaries: Dict[str, Dict[str, float]] = {}
    with result_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            if row.get("video_id") != "SUMMARY":
                continue
            frame_no = row.get("frame_no", "").strip().upper()
            metrics: Dict[str, float] = {}
            for key, value in row.items():
                if key in {"video_id", "frame_no"}:
                    continue
                if value is None or value == "":
                    continue
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue
            summaries[frame_no] = metrics
    return summaries


def infer_metadata(result_path: Path) -> Tuple[str, str, str]:
    parts = result_path.stem.split("__")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {result_path.name}")
    size = parts[0]
    variant = parts[1]
    ckpt = parts[2]
    return size, variant, ckpt


def collect_records(results_dir: Path, preds_dir: Path) -> List[EvalRecord]:
    records: List[EvalRecord] = []
    csv_files = sorted(
        csv_file for csv_file in results_dir.glob("*.csv")
        if "__" in csv_file.stem  # skip summary/aggregate files
    )
    if not csv_files:
        raise SystemExit(f"No CSV files found under {results_dir}")

    for csv_path in csv_files:
        size, variant, ckpt = infer_metadata(csv_path)
        variant_dir = VARIANT_DIR_MAP.get(variant)
        if variant_dir is None:
            raise KeyError(f"Variant '{variant}' not in VARIANT_DIR_MAP")
        pred_dir = preds_dir / size / variant_dir / ckpt
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"Missing prediction directory: {pred_dir}")

        summaries = parse_summary_rows(csv_path)
        micro = summaries.get("MICRO", {})
        macro = summaries.get("MACRO", {})

        node_micro_f1 = micro.get("node_f1", float("nan"))
        edge_micro_f1 = micro.get("edge_f1", float("nan"))
        box_mean_iou = micro.get("box_mean_iou", float("nan"))
        box_iou_0p5_coverage = micro.get("box_iou@0.5_coverage", float("nan"))
        box_iou_0p75_coverage = micro.get("box_iou@0.75_coverage", float("nan"))
        node_macro_f1 = macro.get("node_f1", float("nan"))

        validity_stats = compute_validity_from_directory(pred_dir)
        conformity_stats = compute_conformity_from_directory(pred_dir)

        record = EvalRecord(
            size=size,
            variant=variant,
            ckpt=ckpt,
            result_path=csv_path,
            pred_dir=pred_dir,
            node_micro_f1=node_micro_f1,
            edge_micro_f1=edge_micro_f1,
            box_mean_iou=box_mean_iou,
            box_iou_0p5_coverage=box_iou_0p5_coverage,
            box_iou_0p75_coverage=box_iou_0p75_coverage,
            validity_rate=validity_stats.get("validity_rate", float("nan")),
            conformity_rate=conformity_stats.get("conformity_rate_total", float("nan")),
            node_macro_f1=node_macro_f1,
            extra={
                "edge_macro_f1": macro.get("edge_f1", float("nan")),
                "box_macro_mean_iou": macro.get("box_mean_iou", float("nan")),
                "box_macro_iou@0.5_coverage": macro.get("box_iou@0.5_coverage", float("nan")),
                "box_macro_iou@0.75_coverage": macro.get("box_iou@0.75_coverage", float("nan")),
            },
        )
        records.append(record)
    return records


def mark_winners(records: List[EvalRecord]) -> None:
    """Mark the best checkpoint per (size, variant)."""
    by_key: Dict[Tuple[str, str], List[EvalRecord]] = {}
    for record in records:
        by_key.setdefault((record.size, record.variant), []).append(record)

    for (size, variant), candidates in by_key.items():
        ranked = sorted(candidates, key=lambda r: r.ranking_key(), reverse=True)
        for idx, rec in enumerate(ranked):
            rec.winner = idx == 0


def write_summary(records: Iterable[EvalRecord], output_path: Path) -> None:
    fieldnames = [
        "size",
        "variant",
        "ckpt",
        "node_micro_f1",
        "edge_micro_f1",
        "box_mean_iou",
        "box_iou@0.5_coverage",
        "box_iou@0.75_coverage",
        "validity_rate",
        "conformity_rate",
        "node_macro_f1",
        "winner",
        "result_path",
        "pred_dir",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "size": rec.size,
                    "variant": rec.variant,
                    "ckpt": rec.ckpt,
                    "node_micro_f1": f"{rec.node_micro_f1:.6f}" if not math.isnan(rec.node_micro_f1) else "",
                    "edge_micro_f1": f"{rec.edge_micro_f1:.6f}" if not math.isnan(rec.edge_micro_f1) else "",
                    "box_mean_iou": f"{rec.box_mean_iou:.6f}" if not math.isnan(rec.box_mean_iou) else "",
                    "box_iou@0.5_coverage": f"{rec.box_iou_0p5_coverage:.2f}" if not math.isnan(rec.box_iou_0p5_coverage) else "",
                    "box_iou@0.75_coverage": f"{rec.box_iou_0p75_coverage:.2f}" if not math.isnan(rec.box_iou_0p75_coverage) else "",
                    "validity_rate": f"{rec.validity_rate:.2f}" if not math.isnan(rec.validity_rate) else "",
                    "conformity_rate": f"{rec.conformity_rate:.2f}" if not math.isnan(rec.conformity_rate) else "",
                    "node_macro_f1": f"{rec.node_macro_f1:.6f}" if not math.isnan(rec.node_macro_f1) else "",
                    "winner": "WINNER" if rec.winner else "",
                    "result_path": rec.result_path.as_posix(),
                    "pred_dir": rec.pred_dir.as_posix(),
                }
            )


def print_winners(records: Iterable[EvalRecord]) -> None:
    grouped: Dict[Tuple[str, str], List[EvalRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.size, rec.variant), []).append(rec)

    print(
        "Winner selection (node_micro_f1, edge_micro_f1, box_mean_iou, box_iou@0.5_coverage, "
        "box_iou@0.75_coverage, validity_rate, conformity_rate, node_macro_f1):"
    )
    for (size, variant), candidates in sorted(grouped.items()):
        winner = next((r for r in candidates if r.winner), None)
        if not winner:
            continue
        metrics = (
            winner.node_micro_f1,
            winner.edge_micro_f1,
            winner.box_mean_iou,
            winner.box_iou_0p5_coverage,
            winner.box_iou_0p75_coverage,
            winner.validity_rate,
            winner.conformity_rate,
            winner.node_macro_f1,
        )
        formatted_metrics = []
        for idx, value in enumerate(metrics):
            if math.isnan(value):
                formatted_metrics.append("nan")
            elif idx < 3:
                formatted_metrics.append(f"{value:.4f}")
            else:
                formatted_metrics.append(f"{value:.2f}")
        metric_str = ", ".join(formatted_metrics)
        print(f"  {size}/{variant}: {winner.ckpt} -> {metric_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best validation_dev checkpoints per variant")
    parser.add_argument("--results-dir", type=Path, default=Path("results_valdev"), help="Directory with eval CSV files")
    parser.add_argument("--preds-dir", type=Path, default=Path("preds_valdev"), help="Root directory of prediction outputs")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path for summary table")
    args = parser.parse_args()

    records = collect_records(args.results_dir, args.preds_dir)
    mark_winners(records)

    def sort_key(record: EvalRecord) -> Tuple[str, str, int, str]:
        ckpt_rank = CKPT_ORDER.get(record.ckpt, len(CKPT_ORDER))
        return (record.size, record.variant, ckpt_rank, record.ckpt)

    sorted_records = sorted(records, key=sort_key)
    output_path = args.out or (args.results_dir / "valdev_summary.csv")
    write_summary(sorted_records, output_path)

    print_winners(sorted_records)
    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
