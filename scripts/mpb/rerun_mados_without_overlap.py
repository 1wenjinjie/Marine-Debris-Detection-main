"""Re-evaluate frozen MADOS checkpoints after explicitly excluding confirmed overlap crops or scenes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ORIGINAL_ROOT = ROOT / "runs_mpb" / "external_mados"
OUTPUT_ROOT = ROOT / "runs_mpb" / "overlap_audit" / "filtered_eval"
BASELINE_ROOT = ROOT / "runs_mpb" / "overlap_audit" / "current_baseline"
MATCHES_PATH = ROOT / "runs_mpb" / "overlap_audit" / "all_test" / "image_matches.json"
SUMMARY_PATH = ORIGINAL_ROOT / "summary.json"
PROTOCOL_PATH = ORIGINAL_ROOT / "external_protocol.json"
CELLS = {"full_scene_6ch": 6, "full_scene_ndwi7": 7}
CROP_PATTERN = re.compile(r"^(?:Scene_)?(\d+)_(\d+)$", re.IGNORECASE)
SCENE_PATTERN = re.compile(r"^(?:Scene_)?(\d+)$", re.IGNORECASE)
PLAN_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_exact(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Existing output differs; choose a new --name: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write_exact(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _parse_ids(raw: str | None, pattern: re.Pattern, kind: str) -> list[str]:
    if not raw:
        return []
    result = set()
    for token in raw.split(","):
        token = token.strip()
        match = pattern.fullmatch(token)
        if match is None:
            raise ValueError(f"Invalid {kind} ID: {token!r}")
        result.add("Scene_" + "_".join(str(int(part)) for part in match.groups()))
    return sorted(result, key=lambda item: tuple(map(int, item.removeprefix("Scene_").split("_"))))


def _crop_id(image: Path) -> str:
    match = CROP_PATTERN.fullmatch(image.stem)
    if match is None:
        raise RuntimeError(f"Unexpected MADOS crop name: {image}")
    return f"Scene_{int(match.group(1))}_{int(match.group(2))}"


def _scene_id(crop_id: str) -> str:
    return crop_id.rsplit("_", 1)[0]


def _load_confirmations(test_ids: set[str]) -> tuple[set[str], dict[str, list[str]]]:
    audit = _json_load(MATCHES_PATH)
    official_test_path = ROOT / "data" / "MADOS" / "splits" / "test_X.txt"
    official_ids = [line.strip() for line in official_test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(official_ids) != 728 or len(set(official_ids)) != 728 or set(official_ids) != test_ids:
        raise RuntimeError("Materialized MADOS test IDs differ from the official 728-crop split")
    rows = audit["rows"]
    row_ids = [row["mados_crop"] for row in rows]
    inputs = audit["inputs"]
    not_audited = audit["not_audited"]
    if (inputs.get("mados_official_test_crops") != 728
            or inputs.get("mados_crops_audited") != 728
            or len(rows) != 728 or len(set(row_ids)) != 728 or set(row_ids) != test_ids
            or not_audited.get("mados_test_crops_not_audited") != []):
        raise RuntimeError("Image-match audit does not cover all 728 official MADOS test crops")
    if any(row.get("mados_split") != "test"
           or row.get("mados_scene") != _scene_id(row["mados_crop"])
           or row.get("status") not in {"confirmed_same_imagery", "inconclusive"}
           for row in rows):
        raise RuntimeError("Image-match audit has an invalid split, scene or review status")
    confirmed = {
        row["mados_crop"] for row in rows if row["status"] == "confirmed_same_imagery"
    }
    inconclusive = {row["mados_crop"] for row in rows if row["status"] == "inconclusive"}
    if (confirmed != set(audit["results"]["confirmed_crops"])
            or inconclusive != set(audit["results"]["inconclusive_crops"])):
        raise RuntimeError("Image-match rows disagree with result summary")
    by_scene: dict[str, list[str]] = {}
    for crop_id in sorted(confirmed):
        by_scene.setdefault(_scene_id(crop_id), []).append(crop_id)
    return confirmed, by_scene


def _load_original() -> tuple[dict, dict]:
    summary = _json_load(SUMMARY_PATH)
    protocol = _json_load(PROTOCOL_PATH)
    contract = summary["input_contract"]
    contract_sha = _canonical_sha256(contract)
    if summary.get("input_contract_sha256") != contract_sha or protocol.get("input_contract_sha256") != contract_sha:
        raise RuntimeError("Original MADOS summary/protocol input contracts disagree")
    if protocol.get("input_contract") != contract:
        raise RuntimeError("Original MADOS protocol differs from the summary")
    decision_lock = Path(contract["decision_lock"])
    if _sha256(decision_lock) != contract["decision_lock_sha256"]:
        raise RuntimeError(f"Original MADOS decision lock changed: {decision_lock}")
    current_materialization = _json_load(Path(contract["materialization_manifest"]))
    test_stats = current_materialization["composition_min4"]["test"]
    if (test_stats["crops"] != contract["test_crops"]
            or test_stats["unique_scenes"] != contract["test_unique_scenes"]
            or test_stats["marine_debris_instances_min4"] != contract["test_marine_debris_instances_min4"]
            or test_stats["marine_debris_positive_tiles_min4"] != contract["test_marine_debris_positive_tiles_min4"]):
        raise RuntimeError("Current materialization changes the original MADOS test composition")
    sources = {
        "external_runner": ROOT / "scripts" / "mpb" / "run_mados_external_validation.py",
        "materializer": ROOT / "scripts" / "mpb" / "materialize_mados_external.py",
        "evaluator": ROOT / "scripts" / "v2_evaluate.py",
    }
    for key, path in sources.items():
        if _sha256(path) != contract["source_code_sha256"][key]:
            raise RuntimeError(f"Original MADOS evaluation source changed: {path}")
    lock = _json_load(Path(contract["decision_lock"]))
    if lock.get("test_used_for_decisions") is not False:
        raise RuntimeError("Original MADOS decisions were not validation-only")
    decisions = {(item["cell"], int(item["seed"])): item for item in lock["decisions"]}
    if contract.get("cells") != list(CELLS) or contract.get("seeds") != [0, 1, 2]:
        raise RuntimeError("Unexpected original MADOS cell/seed contract")
    for cell, channels in CELLS.items():
        rows = summary["cells"][cell]["per_seed"]
        if sorted(int(row["seed"]) for row in rows) != [0, 1, 2]:
            raise RuntimeError(f"Missing original seed result: {cell}")
        for row in rows:
            decision = decisions[(cell, int(row["seed"]))]
            weights = Path(row["weights"])
            if (row["input_contract_sha256"] != contract_sha
                    or row["weights"] != decision["weights"]
                    or row["weights_sha256"] != decision["weights_sha256"]
                    or row["validation_locked_balanced_threshold"] != decision["balanced_threshold"]
                    or _sha256(weights) != row["weights_sha256"]
                    or row["metrics_ap"]["channels"] != channels
                    or row["metrics_ap"]["weights"] != row["weights"]):
                raise RuntimeError(f"Original frozen result/weight mismatch: {cell} seed {row['seed']}")
            metrics = row["metrics_ap"]
            if (row["split"] != "test" or row["conf_for_ap"] != 0.001 or row["iou"] != 0.6
                    or row["imgsz"] != 512 or metrics["split"] != "test"
                    or metrics["conf"] != 0.001 or metrics["iou"] != 0.6
                    or metrics["imgsz"] != 512 or metrics["workers"] != 0):
                raise RuntimeError(f"Original evaluation parameters differ: {cell} seed {row['seed']}")
    return summary, protocol


def _load_cell_dataset(original_rows: list[dict], channels: int) -> tuple[Path, Path, dict, dict[str, Path]]:
    config_path = Path(original_rows[0]["metrics_ap"]["data"])
    if any(Path(row["metrics_ap"]["data"]) != config_path for row in original_rows):
        raise RuntimeError(f"Original seeds use different dataset configs: {channels}ch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("channels") != channels:
        raise RuntimeError(f"Dataset channel count changed: {config_path}")
    dataset_root = Path(config["path"])
    test_path = dataset_root / config["test"]
    images = {}
    for raw in test_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        image = Path(raw.strip())
        crop_id = _crop_id(image)
        if crop_id in images or not image.is_file():
            raise RuntimeError(f"Duplicate or missing test image: {image}")
        images[crop_id] = image
    return config_path, test_path, config, images


def _label_path(image: Path) -> Path:
    if image.parent.name != "images":
        raise RuntimeError(f"Unexpected MADOS image layout: {image}")
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def _debris_count(label: Path) -> int:
    if not label.is_file():
        raise FileNotFoundError(label)
    return sum(line.split()[0] == "0" for line in label.read_text(encoding="utf-8").splitlines() if line.strip())


def _ap(metrics: dict) -> dict:
    return {
        "mAP50": metrics["mAP50"],
        "mAP50_95": metrics["mAP50_95"],
        "per_class": {
            name: {"AP50": values["AP50"], "AP50_95": values["AP50_95"]}
            for name, values in metrics["per_class"].items()
        },
    }


def _delta(before: dict, after: dict) -> dict:
    return {
        "mAP50": after["mAP50"] - before["mAP50"],
        "mAP50_95": after["mAP50_95"] - before["mAP50_95"],
        "per_class": {
            name: {
                field: (after["per_class"][name][field] - values[field]
                        if values[field] is not None and after["per_class"][name][field] is not None else None)
                for field in ("AP50", "AP50_95")
            }
            for name, values in before["per_class"].items()
        },
    }


def _mean_ap(rows: list[dict], key: str) -> dict:
    def mean(values: list[float | None]) -> float | None:
        present = [float(value) for value in values if value is not None]
        return sum(present) / len(present) if present else None

    first = rows[0][key]
    return {
        "mAP50": mean([row[key]["mAP50"] for row in rows]),
        "mAP50_95": mean([row[key]["mAP50_95"] for row in rows]),
        "per_class": {
            name: {
                field: mean([row[key]["per_class"][name][field] for row in rows])
                for field in ("AP50", "AP50_95")
            }
            for name in first["per_class"]
        },
    }


def prepare(name: str, exclude_crops: list[str], exclude_scenes: list[str], device: str = "0") -> tuple[dict, Path]:
    if not PLAN_PATTERN.fullmatch(name):
        raise ValueError("--name must use letters, digits, underscores or hyphens (1-80 characters)")
    if not exclude_crops and not exclude_scenes:
        raise ValueError("Provide at least one confirmed crop or scene to exclude")
    original, _ = _load_original()
    datasets = {}
    for cell, channels in CELLS.items():
        datasets[cell] = _load_cell_dataset(original["cells"][cell]["per_seed"], channels)
    reference_ids = set(datasets["full_scene_6ch"][3])
    if reference_ids != set(datasets["full_scene_ndwi7"][3]):
        raise RuntimeError("6ch and 7ch MADOS test crops differ")
    available_scenes = {_scene_id(crop_id) for crop_id in reference_ids}
    unknown_crops = set(exclude_crops) - reference_ids
    unknown_scenes = set(exclude_scenes) - available_scenes
    if unknown_crops or unknown_scenes:
        raise ValueError(f"Exclusion IDs absent from official test: crops={sorted(unknown_crops)}, scenes={sorted(unknown_scenes)}")
    confirmed, confirmed_by_scene = _load_confirmations(reference_ids)
    unconfirmed_crops = set(exclude_crops) - confirmed
    unsupported_scenes = set(exclude_scenes) - set(confirmed_by_scene)
    if unconfirmed_crops or unsupported_scenes:
        raise ValueError(
            f"Exclusions lack confirmed_same_imagery evidence: crops={sorted(unconfirmed_crops)}, scenes={sorted(unsupported_scenes)}"
        )
    removed = sorted(crop_id for crop_id in reference_ids
                     if crop_id in exclude_crops or _scene_id(crop_id) in exclude_scenes)
    kept = reference_ids - set(removed)
    if not kept:
        raise ValueError("Exclusion removes every MADOS test crop")
    labels = {}
    label_fingerprints = []
    for crop_id in reference_ids:
        label_6 = _label_path(datasets["full_scene_6ch"][3][crop_id])
        label_7 = _label_path(datasets["full_scene_ndwi7"][3][crop_id])
        if not label_6.is_file() or not label_7.is_file():
            raise RuntimeError(f"Missing MADOS label: {crop_id}")
        label_sha = _sha256(label_6)
        if label_sha != _sha256(label_7):
            raise RuntimeError(f"6ch and 7ch labels differ: {crop_id}")
        label_fingerprints.append((crop_id, label_sha))
        labels[crop_id] = _debris_count(label_6)
    if len(reference_ids) != original["input_contract"]["test_crops"] or sum(labels.values()) != original["input_contract"]["test_marine_debris_instances_min4"]:
        raise RuntimeError("Current MADOS test composition differs from original summary")
    if len(available_scenes) != original["input_contract"]["test_unique_scenes"]:
        raise RuntimeError("Current MADOS test scene count differs from original summary")
    image_hashes = {
        cell: {crop_id: _sha256(images[crop_id]) for crop_id in sorted(reference_ids)}
        for cell, (_, _, _, images) in datasets.items()
    }
    full_image_fingerprints = {
        cell: _canonical_sha256(sorted(hashes.items())) for cell, hashes in image_hashes.items()
    }
    retained_image_fingerprints = {
        cell: _canonical_sha256([(crop_id, hashes[crop_id]) for crop_id in sorted(kept)])
        for cell, hashes in image_hashes.items()
    }
    source_manifests_sha = {cell: _sha256(dataset[1]) for cell, dataset in datasets.items()}
    source_configs_sha = {cell: _sha256(dataset[0]) for cell, dataset in datasets.items()}
    source_stats_sha = {
        cell: _sha256(Path(dataset[2]["path"]) / f"channel_stats_{CELLS[cell]}ch.json")
        for cell, dataset in datasets.items()
    }
    weights_sha = {
        cell: {str(row["seed"]): row["weights_sha256"] for row in original["cells"][cell]["per_seed"]}
        for cell in CELLS
    }
    eval_contract = {"split": "test", "conf": 0.001, "iou": 0.6, "imgsz": 512, "batch": 8, "workers": 0, "device": device, "allow_test": True}
    baseline_contract = {
        "current_materialization_manifest_sha256": _sha256(Path(original["input_contract"]["materialization_manifest"])),
        "source_manifests_sha256": source_manifests_sha,
        "source_configs_sha256": source_configs_sha,
        "source_channel_stats_sha256": source_stats_sha,
        "source_test_labels_sha256": _canonical_sha256(sorted(label_fingerprints)),
        "full_test_images_sha256": full_image_fingerprints,
        "locked_weights_sha256": weights_sha,
        "evaluator_sha256": _sha256(ROOT / "scripts" / "v2_evaluate.py"),
        "eval": eval_contract,
    }
    baseline_sha = _canonical_sha256(baseline_contract)
    baseline_dir = BASELINE_ROOT / baseline_sha
    out_dir = OUTPUT_ROOT / name
    contract = {
        "original_summary_sha256": _sha256(SUMMARY_PATH),
        "original_protocol_sha256": _sha256(PROTOCOL_PATH),
        "original_input_contract_sha256": original["input_contract_sha256"],
        "original_materialization_manifest_sha256": original["input_contract"]["materialization_manifest_sha256"],
        "current_materialization_manifest_sha256": _sha256(Path(original["input_contract"]["materialization_manifest"])),
        "exclusion": {"crops": exclude_crops, "scenes": exclude_scenes},
        "removed_crop_ids": removed,
        "current_baseline_contract_sha256": baseline_sha,
        "image_matches_sha256": _sha256(MATCHES_PATH),
        "source_manifests_sha256": source_manifests_sha,
        "source_configs_sha256": source_configs_sha,
        "source_channel_stats_sha256": source_stats_sha,
        "source_test_labels_sha256": baseline_contract["source_test_labels_sha256"],
        "retained_test_images_sha256": retained_image_fingerprints,
        "locked_weights_sha256": weights_sha,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "eval": eval_contract,
    }
    contract_sha = _canonical_sha256(contract)
    protocol_path = out_dir / "protocol.json"
    if protocol_path.exists() and _json_load(protocol_path).get("input_contract_sha256") != contract_sha:
        raise RuntimeError(f"Existing plan has a different input contract: {out_dir}")
    baseline_summaries = {}
    for cell, (config_path, _, config, images) in datasets.items():
        manifest_path = baseline_dir / cell / "test.txt"
        manifest = "".join(f"{images[crop_id]}\n" for crop_id in sorted(reference_ids))
        _write_exact(manifest_path, manifest)
        baseline_config = dict(config)
        baseline_config["test"] = str(manifest_path)
        yaml_path = baseline_dir / cell / "dataset.yaml"
        _write_exact(yaml_path, yaml.safe_dump(baseline_config, sort_keys=False, allow_unicode=True))
        baseline_summaries[cell] = {
            "source_config": str(config_path),
            "dataset_config": str(yaml_path),
            "test_manifest": str(manifest_path),
            "test_manifest_sha256": _sha256(manifest_path),
            "dataset_config_sha256": _sha256(yaml_path),
        }
    baseline_protocol = {
        "input_contract_sha256": baseline_sha,
        "input_contract": baseline_contract,
        "datasets": baseline_summaries,
        "scope": "Current full MADOS test baseline for paired comparison with explicit overlap exclusions.",
    }
    _write_json(baseline_dir / "protocol.json", baseline_protocol)
    summaries = {}
    for cell, (config_path, _, config, images) in datasets.items():
        manifest_path = out_dir / cell / "test.txt"
        manifest = "".join(f"{images[crop_id]}\n" for crop_id in sorted(kept))
        _write_exact(manifest_path, manifest)
        filtered_config = dict(config)
        filtered_config["test"] = str(manifest_path)
        yaml_path = out_dir / cell / "dataset.yaml"
        _write_exact(yaml_path, yaml.safe_dump(filtered_config, sort_keys=False, allow_unicode=True))
        summaries[cell] = {
            "original_config": str(config_path),
            "filtered_config": str(yaml_path),
            "filtered_manifest": str(manifest_path),
            "filtered_manifest_sha256": _sha256(manifest_path),
            "filtered_config_sha256": _sha256(yaml_path),
        }
    counts = {
        "original_crops": len(reference_ids),
        "remaining_crops": len(kept),
        "excluded_crops": len(removed),
        "original_scenes": len(available_scenes),
        "remaining_scenes": len({_scene_id(crop_id) for crop_id in kept}),
        "excluded_entire_scenes": len(available_scenes - {_scene_id(crop_id) for crop_id in kept}),
        "original_marine_debris_gt": sum(labels.values()),
        "remaining_marine_debris_gt": sum(labels[crop_id] for crop_id in kept),
        "excluded_marine_debris_gt": sum(labels[crop_id] for crop_id in removed),
    }
    protocol = {
        "input_contract_sha256": contract_sha,
        "input_contract": contract,
        "current_baseline": {"directory": str(baseline_dir), "input_contract_sha256": baseline_sha},
        "counts": counts,
        "datasets": summaries,
        "exclusion_evidence": {
            "confirmed_crop_ids": sorted(set(exclude_crops)),
            "conservative_scene_exclusions": {
                scene: {
                    "confirmed_crop_ids": confirmed_by_scene[scene],
                    "other_removed_crops_not_individually_confirmed": sorted(
                        crop_id for crop_id in removed
                        if _scene_id(crop_id) == scene and crop_id not in confirmed
                    ),
                }
                for scene in exclude_scenes
            },
            "interpretation": "Entire-scene removal is a conservative sensitivity analysis; confirmation of one crop does not verify every crop in that scene.",
        },
        "scope": "Paired current-input sensitivity evaluation after explicit, image-confirmed MADOS test exclusions.",
        "original_materialization_manifest_matches_current": (
            contract["original_materialization_manifest_sha256"] == contract["current_materialization_manifest_sha256"]
        ),
        "original_test_bytes_verifiable": False,
    }
    _write_json(protocol_path, protocol)
    return protocol, out_dir


def _evaluate_current_baseline(original: dict, baseline_dir: Path, baseline_protocol: dict, device: str) -> dict:
    from scripts.v2_evaluate import evaluate

    baseline_sha = baseline_protocol["input_contract_sha256"]
    cells = {}
    for cell, channels in CELLS.items():
        rows = []
        data_yaml = baseline_protocol["datasets"][cell]["dataset_config"]
        for original_row in original["cells"][cell]["per_seed"]:
            seed = int(original_row["seed"])
            weights = Path(original_row["weights"])
            if _sha256(weights) != original_row["weights_sha256"]:
                raise RuntimeError(f"Frozen weights changed: {weights}")
            output_path = baseline_dir / cell / f"seed_{seed}_current_ap.json"
            if output_path.exists():
                row = _json_load(output_path)
                metrics = row.get("current_metrics_ap", {})
                if (row.get("input_contract_sha256") != baseline_sha
                        or row.get("cell") != cell or row.get("seed") != seed
                        or row.get("weights_sha256") != original_row["weights_sha256"]
                        or metrics.get("data") != data_yaml or metrics.get("weights") != str(weights)
                        or metrics.get("channels") != channels or metrics.get("split") != "test"
                        or metrics.get("imgsz") != 512 or metrics.get("conf") != 0.001
                        or metrics.get("iou") != 0.6 or metrics.get("workers") != 0
                        or row.get("current_baseline_ap") != _ap(metrics)):
                    raise RuntimeError(f"Existing current baseline differs: {output_path}")
            else:
                metrics = evaluate(
                    weights=str(weights), data_yaml=data_yaml, channels=channels,
                    split="test", imgsz=512, batch=8, device=device, conf=0.001,
                    iou=0.6, workers=0, project=str(baseline_dir / "ultralytics_eval"),
                    name=f"{cell}_s{seed}_current_ap", allow_test=True,
                )
                row = {
                    "input_contract_sha256": baseline_sha,
                    "cell": cell,
                    "seed": seed,
                    "weights": str(weights),
                    "weights_sha256": original_row["weights_sha256"],
                    "current_baseline_ap": _ap(metrics),
                    "current_metrics_ap": metrics,
                }
                _write_json(output_path, row)
            rows.append(row)
        cells[cell] = {"per_seed": rows, "mean_current_baseline_ap": _mean_ap(rows, "current_baseline_ap")}
    summary = {
        "input_contract_sha256": baseline_sha,
        "input_contract": baseline_protocol["input_contract"],
        "datasets": baseline_protocol["datasets"],
        "cells": cells,
    }
    _write_json(baseline_dir / "summary.json", summary)
    return cells


def run(protocol: dict, out_dir: Path, device: str) -> None:
    from scripts.v2_evaluate import evaluate

    original = _json_load(SUMMARY_PATH)
    contract_sha = protocol["input_contract_sha256"]
    if protocol["input_contract"]["eval"]["device"] != device:
        raise RuntimeError("Run device differs from the prepared input contract")
    baseline_dir = Path(protocol["current_baseline"]["directory"])
    baseline_protocol = _json_load(baseline_dir / "protocol.json")
    if baseline_protocol.get("input_contract_sha256") != protocol["current_baseline"]["input_contract_sha256"]:
        raise RuntimeError("Current baseline protocol does not match filtered plan")
    if baseline_protocol["input_contract"]["eval"]["device"] != device:
        raise RuntimeError("Current baseline device differs from the filtered plan")
    baseline_cells = _evaluate_current_baseline(original, baseline_dir, baseline_protocol, device)
    cells = {}
    for cell, channels in CELLS.items():
        results = []
        data_yaml = protocol["datasets"][cell]["filtered_config"]
        for original_row, baseline_row in zip(original["cells"][cell]["per_seed"], baseline_cells[cell]["per_seed"]):
            seed = int(original_row["seed"])
            if baseline_row["seed"] != seed or baseline_row["weights_sha256"] != original_row["weights_sha256"]:
                raise RuntimeError(f"Current baseline seed/weight mismatch: {cell} seed {seed}")
            output_path = out_dir / cell / f"seed_{seed}_filtered_ap.json"
            if output_path.exists():
                row = _json_load(output_path)
                if (row.get("input_contract_sha256") != contract_sha
                        or row.get("current_baseline_contract_sha256") != baseline_protocol["input_contract_sha256"]
                        or row.get("weights_sha256") != original_row["weights_sha256"]
                        or row.get("cell") != cell or row.get("seed") != seed
                        or row.get("current_baseline_ap") != baseline_row["current_baseline_ap"]
                        or row.get("historical_original_ap_reference") != _ap(original_row["metrics_ap"])
                        or row.get("filtered_metrics_ap", {}).get("data") != data_yaml
                        or row.get("filtered_ap") != _ap(row["filtered_metrics_ap"])
                        or row.get("delta_ap_filtered_minus_current_baseline") != _delta(
                            baseline_row["current_baseline_ap"], row["filtered_ap"]
                        )):
                    raise RuntimeError(f"Existing seed result has different inputs: {output_path}")
            else:
                weights = Path(original_row["weights"])
                if _sha256(weights) != original_row["weights_sha256"]:
                    raise RuntimeError(f"Frozen weights changed: {weights}")
                metrics = evaluate(
                    weights=str(weights), data_yaml=data_yaml, channels=channels,
                    split="test", imgsz=512, batch=8, device=device, conf=0.001,
                    iou=0.6, workers=0, project=str(out_dir / "ultralytics_eval"),
                    name=f"{cell}_s{seed}_filtered_ap", allow_test=True,
                )
                before = baseline_row["current_baseline_ap"]
                after = _ap(metrics)
                row = {
                    "input_contract_sha256": contract_sha,
                    "current_baseline_contract_sha256": baseline_protocol["input_contract_sha256"],
                    "cell": cell,
                    "seed": seed,
                    "weights": str(weights),
                    "weights_sha256": original_row["weights_sha256"],
                    "current_baseline_ap": before,
                    "filtered_ap": after,
                    "delta_ap_filtered_minus_current_baseline": _delta(before, after),
                    "historical_original_ap_reference": _ap(original_row["metrics_ap"]),
                    "filtered_metrics_ap": metrics,
                    "excluded_counts": protocol["counts"],
                }
                _write_json(output_path, row)
            results.append(row)
            print(f"{cell} seed={seed}: current MD AP50={row['current_baseline_ap']['per_class']['Marine_Debris']['AP50']}, filtered MD AP50={row['filtered_ap']['per_class']['Marine_Debris']['AP50']}")
        cells[cell] = {
            "per_seed": results,
            "mean_current_baseline_ap": _mean_ap(results, "current_baseline_ap"),
            "mean_filtered_ap": _mean_ap(results, "filtered_ap"),
            "mean_delta_ap_filtered_minus_current_baseline": _mean_ap(results, "delta_ap_filtered_minus_current_baseline"),
            "mean_historical_original_ap_reference": _mean_ap(results, "historical_original_ap_reference"),
        }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_contract_sha256": contract_sha,
        "input_contract": protocol["input_contract"],
        "current_baseline": protocol["current_baseline"],
        "counts": protocol["counts"],
        "datasets": protocol["datasets"],
        "exclusion_evidence": protocol["exclusion_evidence"],
        "historical_original_reference": {
            "summary": str(SUMMARY_PATH),
            "locked_materialization_manifest_sha256": protocol["input_contract"]["original_materialization_manifest_sha256"],
            "current_materialization_manifest_sha256": protocol["input_contract"]["current_materialization_manifest_sha256"],
            "manifest_drift": not protocol["original_materialization_manifest_matches_current"],
            "interpretation": "Historical AP is descriptive only; the primary delta compares current full-test and filtered-test runs.",
        },
        "cells": cells,
    }
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        _write_json(summary_path, summary)
    existing = _json_load(summary_path)
    if any(existing.get(key) != summary[key] for key in ("input_contract_sha256", "input_contract", "current_baseline", "counts", "datasets", "exclusion_evidence", "historical_original_reference", "cells")):
        raise RuntimeError(f"Existing summary has different inputs or results: {summary_path}")
    print(f"Filtered MADOS evaluation: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Unique output plan name")
    parser.add_argument("--exclude-crops", help="Comma-separated confirmed MADOS crop IDs, e.g. 133_1,140_10")
    parser.add_argument("--exclude-scenes", help="Comma-separated confirmed MADOS scene IDs, e.g. 133,140")
    parser.add_argument("--stage", choices=("prepare", "run"), default="prepare")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    crops = _parse_ids(args.exclude_crops, CROP_PATTERN, "crop")
    scenes = _parse_ids(args.exclude_scenes, SCENE_PATTERN, "scene")
    protocol, out_dir = prepare(args.name, crops, scenes, args.device)
    print(json.dumps({"output": str(out_dir), "counts": protocol["counts"], "input_contract_sha256": protocol["input_contract_sha256"]}, ensure_ascii=False, indent=2))
    if args.stage == "run":
        run(protocol, out_dir, args.device)


if __name__ == "__main__":
    main()
