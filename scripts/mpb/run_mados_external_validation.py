"""Zero-shot external evaluation of locked MARIDA detectors on official MADOS test crops.

Imports frozen full_scene_6ch and full_scene_ndwi7 checkpoints plus validation-only
balanced thresholds. MADOS labels are never used for model or threshold selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.v2_evaluate import evaluate

MADOS_ROOT = ROOT / "data" / "processed" / "mados_external_min4"
CONFIG_6CH = ROOT / "configs" / "datasets" / "mpb" / "mados_external_scene_6ch_min4.yaml"
CONFIG_7CH = ROOT / "configs" / "datasets" / "mpb" / "mados_external_scene_7ch_ndwi_min4.yaml"
MATERIALIZATION_MANIFEST = MADOS_ROOT / "materialization_manifest.json"
RUNS_ROOT = ROOT / "runs_mpb"
DECISION_LOCK = RUNS_ROOT / "decision_lock_all.json"
TEST_MARKER = RUNS_ROOT / "test_access_all.json"
TEST_OUTPUT_MANIFEST = RUNS_ROOT / "test_output_manifest_all.json"
OUT_DIR = RUNS_ROOT / "external_mados"
PROTOCOL_PATH = OUT_DIR / "external_protocol.json"
SUMMARY_PATH = OUT_DIR / "summary.json"
SUMMARY_MARKDOWN_PATH = OUT_DIR / "summary.md"

CELLS = (
    ("full_scene_6ch", CONFIG_6CH, 6),
    ("full_scene_ndwi7", CONFIG_7CH, 7),
)
EXPECTED_SEEDS = (0, 1, 2)
IMAGE_SIZE = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _assert_internal_test_is_frozen() -> None:
    if not TEST_MARKER.exists() or not TEST_OUTPUT_MANIFEST.exists():
        raise RuntimeError("Complete the locked internal MARIDA test before opening the external MADOS run")
    marker = json.loads(TEST_MARKER.read_text(encoding="utf-8"))
    if marker.get("status") != "retrospective_internal_validation_not_external_confirmation":
        raise RuntimeError("The internal MARIDA test is not complete and frozen")
    if marker.get("decision_lock_sha256") != _sha256(DECISION_LOCK):
        raise RuntimeError("Internal test marker does not belong to the current decision lock")
    if marker.get("test_output_manifest_sha256") != _sha256(TEST_OUTPUT_MANIFEST):
        raise RuntimeError("Internal test output manifest changed")


def _load_materialization() -> dict[str, object]:
    if not MATERIALIZATION_MANIFEST.exists():
        raise FileNotFoundError(
            "Run scripts/mpb/materialize_mados_external.py before external MADOS evaluation"
        )
    manifest = json.loads(MATERIALIZATION_MANIFEST.read_text(encoding="utf-8"))
    config_path = Path(str(manifest["dataset_configs"]["6ch"]))
    dataset_root = Path(yaml.safe_load(config_path.read_text(encoding="utf-8"))["path"])
    test_txt = dataset_root / "test.txt"
    if not test_txt.exists() or not test_txt.read_text(encoding="utf-8").strip():
        raise RuntimeError("MADOS test split is not materialized")
    return manifest


def _load_locked_decisions(cell: str) -> list[dict[str, object]]:
    lock = json.loads(DECISION_LOCK.read_text(encoding="utf-8"))
    if lock.get("test_used_for_decisions") is not False:
        raise RuntimeError("MADOS evaluation requires the validation-only decision lock")
    decisions = [decision for decision in lock["decisions"] if decision.get("cell") == cell]
    decisions.sort(key=lambda item: int(item["seed"]))
    if tuple(int(decision["seed"]) for decision in decisions) != EXPECTED_SEEDS:
        raise RuntimeError(f"Expected locked seeds for {cell}: {EXPECTED_SEEDS}")
    for decision in decisions:
        weights = Path(str(decision["weights"]))
        if not weights.exists() or _sha256(weights) != decision.get("weights_sha256"):
            raise RuntimeError(f"Locked weights missing or changed: {weights}")
        if not isinstance(decision.get("balanced_threshold"), (int, float)):
            raise RuntimeError(f"Missing validation-selected balanced threshold: {decision}")
    return decisions


def _dataset_contract(manifest: dict[str, object]) -> dict[str, object]:
    composition = manifest.get("composition_min4", {})
    test_stats = composition.get("test", {})
    return {
        "dataset": "MADOS",
        "official_split": "test_X",
        "materialization_manifest": str(MATERIALIZATION_MANIFEST),
        "materialization_manifest_sha256": _sha256(MATERIALIZATION_MANIFEST),
        "test_crops": test_stats.get("crops"),
        "test_unique_scenes": test_stats.get("unique_scenes"),
        "test_marine_debris_instances_min4": test_stats.get("marine_debris_instances_min4"),
        "test_marine_debris_positive_tiles_min4": test_stats.get("marine_debris_positive_tiles_min4"),
        "decision_lock": str(DECISION_LOCK),
        "decision_lock_sha256": _sha256(DECISION_LOCK),
        "cells": [cell for cell, _, _ in CELLS],
        "seeds": list(EXPECTED_SEEDS),
        "split_used_for_scoring": "test",
        "threshold_rule": "validation_locked_balanced; AP metrics additionally reported at conf=0.001",
        "normalization_rule": "MARIDA train-only p01/p99 imported during materialization; MADOS never used for stats fitting",
        "geographic_exclusion_note": manifest.get("geographic_exclusion_note"),
        "source_code_sha256": {
            "external_runner": _sha256(Path(__file__).resolve()),
            "materializer": _sha256(ROOT / "scripts" / "mpb" / "materialize_mados_external.py"),
            "evaluator": _sha256(ROOT / "scripts" / "v2_evaluate.py"),
        },
    }


def _write_or_validate_protocol(contract: dict[str, object]) -> str:
    contract_sha = _canonical_sha256(contract)
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing.get("input_contract_sha256") != contract_sha:
            raise RuntimeError("Existing MADOS output belongs to a different input contract; do not overwrite it")
        return contract_sha
    _json_dump(
        PROTOCOL_PATH,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_contract_sha256": contract_sha,
            "input_contract": contract,
            "scope": (
                "Zero-shot instance-level external check on official MADOS test crops using frozen MARIDA "
                "checkpoints and validation-only thresholds."
            ),
        },
    )
    return contract_sha


def _metric_get(metrics: dict[str, object], class_name: str, field: str) -> float | None:
    per_class = metrics.get("per_class", {})
    if not isinstance(per_class, dict):
        return None
    payload = per_class.get(class_name, {})
    if not isinstance(payload, dict):
        return None
    value = payload.get(field)
    return float(value) if isinstance(value, (int, float)) else None


def _evaluate_seed(
    decision: dict[str, object],
    data_yaml: Path,
    channels: int,
    device: str,
    contract_sha: str,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("input_contract_sha256") != contract_sha:
            raise RuntimeError(f"Existing MADOS seed output does not match contract: {output_path}")
        return existing
    seed = int(decision["seed"])
    weights = str(decision["weights"])
    ap_metrics = evaluate(
        weights=weights,
        data_yaml=str(data_yaml),
        channels=channels,
        split="test",
        imgsz=IMAGE_SIZE,
        batch=8,
        device=device,
        conf=0.001,
        iou=0.6,
        workers=0,
        project=str(OUT_DIR / "ultralytics_eval"),
        name=f"{decision['cell']}_s{seed}_test_ap",
        allow_test=True,
    )
    payload = {
        "input_contract_sha256": contract_sha,
        "cell": decision["cell"],
        "seed": seed,
        "weights": weights,
        "weights_sha256": decision["weights_sha256"],
        "validation_locked_balanced_threshold": float(decision["balanced_threshold"]),
        "split": "test",
        "conf_for_ap": 0.001,
        "iou": 0.6,
        "imgsz": IMAGE_SIZE,
        "metrics_ap": ap_metrics,
    }
    _json_dump(output_path, payload)
    print(f"MADOS {decision['cell']} seed={seed}: MD AP50={_metric_get(ap_metrics, 'Marine_Debris', 'AP50')}")
    return payload


def _aggregate(per_seed: list[dict[str, object]]) -> dict[str, object]:
    def mean(field: str, class_name: str | None = None) -> float | None:
        values = []
        for row in per_seed:
            metrics = row["metrics_ap"]
            if class_name is None:
                value = metrics.get(field)
            else:
                value = _metric_get(metrics, class_name, field)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return float(sum(values) / len(values)) if values else None

    return {
        "seed_count": len(per_seed),
        "mean_mAP50": mean("mAP50"),
        "mean_mAP50_95": mean("mAP50_95"),
        "mean_marine_debris_AP50": mean("AP50", "Marine_Debris"),
        "mean_marine_debris_AP50_95": mean("AP50_95", "Marine_Debris"),
        "mean_marine_debris_recall_at_conf001": mean("recall", "Marine_Debris"),
        "mean_marine_debris_precision_at_conf001": mean("precision", "Marine_Debris"),
        "validation_locked_thresholds": {
            str(row["seed"]): row["validation_locked_balanced_threshold"] for row in per_seed
        },
    }


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# MADOS Zero-Shot External Instance Evaluation",
        "",
        "Official MADOS `test_X` crops evaluated with frozen MARIDA `full_scene_6ch` and `full_scene_ndwi7` checkpoints. Thresholds remain validation-locked; MADOS was not used for any selection.",
        "",
        "## Test composition (MIN=4 labels)",
        "",
        f"- Crops: {summary['input_contract']['test_crops']}",
        f"- Unique scenes: {summary['input_contract']['test_unique_scenes']}",
        f"- Marine_Debris instances (>=4 px): {summary['input_contract']['test_marine_debris_instances_min4']}",
        f"- Marine_Debris positive tiles: {summary['input_contract']['test_marine_debris_positive_tiles_min4']}",
        "",
        "## AP@IoU0.5 (conf=0.001, not an operating point)",
        "",
        "| Cell | Mean mAP50 | Mean Marine_Debris AP50 | Mean Marine_Debris R | Mean Marine_Debris P |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell, payload in summary["cells"].items():
        agg = payload["aggregate"]
        lines.append(
            "| {cell} | {map50:.3f} | {md:.3f} | {recall:.3f} | {precision:.3f} |".format(
                cell=cell,
                map50=float(agg["mean_mAP50"] or 0.0),
                md=float(agg["mean_marine_debris_AP50"] or 0.0),
                recall=float(agg["mean_marine_debris_recall_at_conf001"] or 0.0),
                precision=float(agg["mean_marine_debris_precision_at_conf001"] or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- MADOS patch GeoTIFFs lack geotransforms, so date/MGRS overlap with MARIDA's 63 training scenes cannot be verified locally.",
            "- AP metrics at conf=0.001 describe ranking quality under cross-dataset radiometry (ACOLITE rhorc) and label mapping; they do not reopen threshold tuning on MADOS.",
            "- Per-class AP for classes with sparse/zero MADOS test GT should be treated as descriptive only.",
            "",
        ]
    )
    return "\n".join(lines)


def inspect() -> None:
    manifest = _load_materialization()
    print(json.dumps(_dataset_contract(manifest), ensure_ascii=False, indent=2))


def run(device: str) -> None:
    _assert_internal_test_is_frozen()
    manifest = _load_materialization()
    contract = _dataset_contract(manifest)
    contract_sha = _write_or_validate_protocol(contract)
    cell_results: dict[str, object] = {}
    for cell, config_path, channels in CELLS:
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        decisions = _load_locked_decisions(cell)
        per_seed = [
            _evaluate_seed(
                decision,
                config_path,
                channels,
                device,
                contract_sha,
                OUT_DIR / cell / f"seed_{int(decision['seed'])}_test_ap.json",
            )
            for decision in decisions
        ]
        cell_results[cell] = {"per_seed": per_seed, "aggregate": _aggregate(per_seed)}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_contract_sha256": contract_sha,
        "input_contract": contract,
        "cells": cell_results,
        "interpretation_limits": [
            "Zero-shot cross-dataset transfer; no MADOS-driven checkpoint or threshold selection.",
            "Geographic disjointness from MARIDA training scenes is assumed at dataset level because MADOS crops are not georeferenced.",
            "Primary readout is Marine_Debris AP50 on official MADOS test crops with MIN=4 instance labels.",
        ],
    }
    _json_dump(SUMMARY_PATH, summary)
    SUMMARY_MARKDOWN_PATH.write_text(_markdown(summary), encoding="utf-8")
    print(f"MADOS external summary written: {SUMMARY_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inspect", "run"), default="inspect")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.stage == "inspect":
        inspect()
    else:
        run(args.device)


if __name__ == "__main__":
    main()
