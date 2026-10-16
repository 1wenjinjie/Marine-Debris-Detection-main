"""Count the preregistered four-cell MADOS/MARIDA detection diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mpb import run_experiments as mpb

OUT = ROOT / "runs_mpb" / "mados_overlap"
PLAN = OUT / "analysis_plan.md"
PLAN_HASH = OUT / "analysis_plan.sha256"
MATCHES = OUT / "matches.csv"
RADIOMETRY = OUT / "radiometry.csv"
T2_SUMMARY = OUT / "t2_summary.json"
DECISIONS = ROOT / "runs_mpb" / "decision_lock_all.json"
INVENTORY = ROOT / "data" / "processed" / "marida_mpb_v1" / "audit" / "patch_inventory.csv"
MARIDA_CONFIG = ROOT / "configs" / "datasets" / "mpb"
MADOS_CONFIG = ROOT / "configs" / "datasets" / "mpb"
CELLS = {"full_scene_6ch": ("scene_6ch_min4.yaml", "mados_external_scene_6ch_min4.yaml"),
         "full_scene_ndwi7": ("scene_7ch_ndwi_min4.yaml", "mados_external_scene_7ch_ndwi_min4.yaml")}
SEEDS = (0, 1, 2)
MIN_CONF = 0.001
MARGIN = 3.0
IOU = 0.5


@lru_cache(maxsize=None)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_plan_hash() -> str:
    expected = PLAN_HASH.read_text(encoding="utf-8").strip().split()[0].lower()
    actual = sha256(PLAN)
    if expected != actual:
        raise RuntimeError(f"analysis_plan.md SHA-256 changed: {expected} != {actual}")
    return actual


def flag(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or str(value).strip().lower() in {"", "nan", "none", "undetermined", "null"}:
        return None
    if str(value).strip().lower() in {"true", "1", "yes"}:
        return True
    if str(value).strip().lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean status: {value}")


def assert_t2_clear(plan_hash: str) -> dict[str, object]:
    summary = json.loads(T2_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("analysis_plan_sha256") != plan_hash:
        raise RuntimeError("T2 summary has a different or missing locked-plan hash")
    if summary.get("stop_signal") is not False:
        raise RuntimeError("T2 has not explicitly cleared its preprocessing stop signal")
    with RADIOMETRY.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("preprocessing_mismatch", "raw_materialization_failure", "input_validation_failure"):
            if key in row and flag(row[key]) is True:
                raise RuntimeError(f"T2 stop signal in radiometry.csv: {key}")
    return summary


def config_paths(config_path: Path, split: str) -> dict[str, Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(str(config["path"]))
    manifest = Path(str(config[split]))
    if not manifest.is_absolute():
        manifest = root / manifest
    paths = [Path(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(paths) != len({path.stem for path in paths}):
        raise RuntimeError(f"Duplicate image stem in {manifest}")
    return {path.stem: path for path in paths}


def label_boxes(image_path: Path, side: int) -> np.ndarray:
    label_path = mpb._label_path(image_path)
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if values and int(float(values[0])) == 0:
            cx, cy, width, height = map(float, values[1:5])
            boxes.append([(cx - width / 2) * side, (cy - height / 2) * side,
                          (cx + width / 2) * side, (cy + height / 2) * side])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def overlap(dy: int, dx: int) -> tuple[float, float, float, float]:
    return (float(max(0, dx)), float(max(0, dy)),
            float(min(256, 240 + dx)), float(min(256, 240 + dy)))


def keep_centers(boxes: np.ndarray, window: tuple[float, float, float, float]) -> np.ndarray:
    centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0
    x0, y0, x1, y1 = window
    return boxes[(centers[:, 0] >= x0 + MARGIN) & (centers[:, 0] <= x1 - MARGIN)
                 & (centers[:, 1] >= y0 + MARGIN) & (centers[:, 1] <= y1 - MARGIN)]


def transform(boxes: np.ndarray, native_side: int, dy: int = 0, dx: int = 0) -> np.ndarray:
    result = boxes.astype(np.float32, copy=True).reshape(-1, boxes.shape[-1])
    result[:, :4] *= np.float32(native_side / 512.0)
    result[:, [0, 2]] += dx
    result[:, [1, 3]] += dy
    return result


def read_pairs() -> tuple[list[dict[str, object]], dict[str, object]]:
    with INVENTORY.open(newline="", encoding="utf-8-sig") as handle:
        inventory = {row["stem"]: row for row in csv.DictReader(handle)}
    with RADIOMETRY.open(newline="", encoding="utf-8-sig") as handle:
        rad = {}
        for row in csv.DictReader(handle):
            key = (row["mados_id"], Path(row["marida_patch"]).stem)
            if key in rad:
                raise RuntimeError(f"Duplicate T2 radiometry key: {key}")
            value = row.get("same_acq_same_processing", row.get("same_acq"))
            rad[key] = flag(value)
    with MATCHES.open(newline="", encoding="utf-8-sig") as handle:
        source = list(csv.DictReader(handle))
    pairs = []
    rejected = defaultdict(int)
    for row in source:
        if row.get("alignment_status", "").lower() == "alignment_unresolved":
            rejected["alignment_unresolved"] += 1
            continue
        try:
            dy, dx = int(row["dy"]), int(row["dx"])
            score = float(row["match_score"])
        except (ValueError, KeyError, TypeError):
            rejected["missing_integer_alignment"] += 1
            continue
        marida_id = Path(row["marida_patch"]).stem
        if score < 0.95 or int(row["n_overlap_px"]) < 10000:
            rejected["outside_locked_t1_gate"] += 1
            continue
        if marida_id not in inventory:
            raise RuntimeError(f"Patch absent from inventory: {marida_id}")
        split = inventory[marida_id]["scene_split"]
        if split != row["marida_split"]:
            raise RuntimeError(f"MARIDA split conflict for {marida_id}")
        window = overlap(dy, dx)
        if (window[2] - window[0]) * (window[3] - window[1]) < 10000:
            raise RuntimeError(f"T1 overlap mismatch for {row['mados_id']}, {marida_id}")
        pairs.append({"mados_id": row["mados_id"], "marida_patch": marida_id,
                      "marida_split": split, "dy": dy, "dx": dx,
                      "match_score": score, "same_acq": rad.get((row["mados_id"], marida_id)),
                      "window": window})
    if not pairs:
        raise RuntimeError("No valid T1 pairs")
    return pairs, {"source_pairs": len(source), "valid_pairs": len(pairs), "rejected": dict(rejected)}


def select_primary(pairs: list[dict[str, object]], mados_paths: dict[str, Path]) -> tuple[list[dict[str, object]], dict[str, int]]:
    grouped = defaultdict(list)
    for row in pairs:
        grouped[str(row["mados_id"])].append(row)
    selected = []
    all_gt = 0
    for crop_id, candidates in sorted(grouped.items()):
        if crop_id not in mados_paths:
            raise RuntimeError(f"MADOS test crop absent from manifest: {crop_id}")
        gt = label_boxes(mados_paths[crop_id], 240)
        all_gt += len(gt)
        for row in candidates:
            shifted = gt.copy()
            shifted[:, [0, 2]] += int(row["dx"])
            shifted[:, [1, 3]] += int(row["dy"])
            x0, y0, x1, y1 = row["window"]
            ctr = (shifted[:, :2] + shifted[:, 2:]) / 2
            row["gt_centers_covered"] = int(((ctr[:, 0] >= x0) & (ctr[:, 0] <= x1)
                                              & (ctr[:, 1] >= y0) & (ctr[:, 1] <= y1)).sum())
        chosen = sorted(candidates, key=lambda item: (-int(item["gt_centers_covered"]),
                              -float(item["match_score"]), str(item["marida_patch"])))[0]
        chosen["mados_gt_all"] = len(gt)
        selected.append(chosen)
    return selected, {"matched_crops": len(grouped), "matched_mados_gt": all_gt,
                      "primary_pairs": len(selected)}


def decisions() -> dict[tuple[str, int], dict[str, object]]:
    payload = json.loads(DECISIONS.read_text(encoding="utf-8"))
    if payload.get("test_used_for_decisions") is not False:
        raise RuntimeError("Expected validation-only threshold lock")
    chosen = {}
    for row in payload["decisions"]:
        key = (str(row["cell"]), int(row["seed"]))
        if key[0] in CELLS and key[1] in SEEDS:
            weight = Path(str(row["weights"]))
            if sha256(weight) != row["weights_sha256"]:
                raise RuntimeError(f"Locked weight changed: {weight}")
            chosen[key] = row
    expected = {(cell, seed) for cell in CELLS for seed in SEEDS}
    if set(chosen) != expected:
        raise RuntimeError(f"Missing locked models: {expected - set(chosen)}")
    return chosen


def old_ledger(run_dir: Path, split: str, decision: dict[str, object]) -> tuple[dict[str, dict[str, object]], str]:
    name = "val_predictions.json" if split == "val" else "test_predictions_locked.json"
    path = run_dir / name
    actual = sha256(path)
    if split == "val":
        expected = str(decision["validation_ledger_sha256"])
    else:
        result = json.loads((run_dir / "eval_test_locked.json").read_text(encoding="utf-8"))
        expected = str(result["prediction_ledger_sha256"])
    if actual != expected:
        raise RuntimeError(f"Historical {split} prediction ledger changed: {path}")
    entries = mpb._read_prediction_ledger(path)
    return {str(entry["stem"]): {
        "predictions_xyxy_conf_cls": np.asarray(entry["predictions"]).tolist(),
        "ground_truth_xyxy": np.asarray(entry["ground_truth"]).tolist(),
    } for entry in entries}, actual


def infer_missing(cell: str, seed: int, decision: dict[str, object], paths: dict[str, dict[str, Path]],
                  selected: list[dict[str, object]], device: str, config_hashes: dict[str, str]) -> tuple[dict[str, dict[str, dict[str, object]]], Path]:
    from ultralytics import YOLO
    import torch

    target_ids = {"marida_train": sorted({str(row["marida_patch"]) for row in selected if row["marida_split"] == "train"}),
                  "mados": sorted({str(row["mados_id"]) for row in selected})}
    inputs = {}
    for kind, ids in target_ids.items():
        mapping = paths["marida_train" if kind == "marida_train" else "mados"]
        for stem in ids:
            if stem not in mapping:
                raise RuntimeError(f"Missing {kind} image: {stem}")
            inputs[f"{kind}:{stem}"] = sha256(mapping[stem])
    cache_path = OUT / "t4_predictions" / f"{cell}_s{seed}.json"
    contract = {"format": "mados-marida-t4-inference-v1", "cell": cell, "seed": seed,
                "plan_sha256": locked_plan_hash(), "weight_sha256": decision["weights_sha256"],
                "input_sha256": inputs, "config_and_stats_sha256": config_hashes,
                "imgsz": 512, "conf": MIN_CONF, "nms_iou": 0.60, "max_det": 300}
    if cache_path.exists():
        previous = json.loads(cache_path.read_text(encoding="utf-8"))
        if previous.get("contract") != contract:
            raise RuntimeError(f"Existing T4 cache has different inputs: {cache_path}")
        return previous["entries"], cache_path

    native_device = torch.device("cpu" if device == "cpu" or not torch.cuda.is_available()
                                 else f"cuda:{device.split(',')[0]}")
    model = YOLO(str(decision["weights"])).model.to(native_device).eval()
    nc = int(model.yaml.get("nc", 3))
    entries = {"marida_train": {}, "mados": {}}
    for kind, ids in target_ids.items():
        config = paths["marida_config" if kind == "marida_train" else "mados_config"]
        dataset_root = Path(str(yaml.safe_load(config.read_text(encoding="utf-8"))["path"]))
        channels = 6 if cell == "full_scene_6ch" else 7
        low, high = mpb._load_stats(dataset_root, channels)
        mapping = paths["marida_train" if kind == "marida_train" else "mados"]
        for index, stem in enumerate(ids, 1):
            array, geometry = mpb._preprocess(mapping[stem], low, high, 512)
            tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0).to(native_device)
            prediction = mpb._raw_predict(model, tensor, nc)
            prediction = prediction[prediction[:, 5].astype(int) == 0]
            gt = mpb._load_gt(mpb._label_path(mapping[stem]), geometry)
            entries[kind][stem] = {"predictions_xyxy_conf_cls": prediction.tolist(),
                                   "ground_truth_xyxy": gt.tolist()}
            if index % 50 == 0 or index == len(ids):
                print(f"{cell} s{seed}: {kind} {index}/{len(ids)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"contract": contract, "entries": entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entries, cache_path


def as_pred(entry: dict[str, object]) -> np.ndarray:
    return np.asarray(entry["predictions_xyxy_conf_cls"], dtype=np.float32).reshape(-1, 6)


def as_gt(entry: dict[str, object]) -> np.ndarray:
    return np.asarray(entry["ground_truth_xyxy"], dtype=np.float32).reshape(-1, 4)


def verify_gt(entry: dict[str, object], image_path: Path, native_side: int) -> None:
    recorded = transform(as_gt(entry), native_side)[:, :4]
    current = label_boxes(image_path, native_side)
    if recorded.shape != current.shape:
        raise RuntimeError(f"Historical ledger GT count differs from current labels: {image_path}")
    if len(recorded):
        order_recorded = np.lexsort(recorded.T[::-1])
        order_current = np.lexsort(current.T[::-1])
        if not np.allclose(recorded[order_recorded], current[order_current], atol=0.001, rtol=0):
            raise RuntimeError(f"Historical ledger GT differs from current labels: {image_path}")


def count(pred: np.ndarray, gt: np.ndarray, threshold: float) -> tuple[int, int, int]:
    pred = pred[pred[:, 4] >= threshold]
    order = np.argsort(-pred[:, 4], kind="stable")
    pred = pred[order]
    matrix = mpb._iou_matrix(pred[:, :4], gt)
    hit = set()
    tp = 0
    for pi in range(len(pred)):
        for gi in np.argsort(-matrix[pi]):
            if matrix[pi, gi] < IOU:
                break
            if int(gi) not in hit:
                hit.add(int(gi))
                tp += 1
                break
    return tp, len(pred) - tp, len(gt) - tp


def score_pair(pair: dict[str, object], mar: dict[str, object], mad: dict[str, object],
               threshold: float) -> list[dict[str, object]]:
    dy, dx = int(pair["dy"]), int(pair["dx"])
    window = pair["window"]
    pm = keep_centers(transform(as_pred(mar), 256), window)
    pd = keep_centers(transform(as_pred(mad), 240, dy, dx), window)
    gm = keep_centers(transform(as_gt(mar), 256), window)
    gd = keep_centers(transform(as_gt(mad), 240, dy, dx), window)
    rows = []
    for grid, pred, gt in (("A", pm, gm), ("B", pm, gd), ("C", pd, gm), ("D", pd, gd)):
        tp, fp, fn = count(pred, gt, threshold)
        rows.append({"grid": grid, "tp": tp, "fp": fp, "fn": fn, "hit_gt": tp,
                     "gt": len(gt), "pred": int((pred[:, 4] >= threshold).sum())})
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite T4 result: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    plan_hash = locked_plan_hash()
    t2_summary = assert_t2_clear(plan_hash)
    result_paths = [OUT / name for name in ("t4_candidate_pairs.csv", "t4_primary_pairs.csv",
                    "t4_pair_counts.csv", "t4_summary.csv", "t4_provenance.json")]
    occupied = [path for path in result_paths if path.exists()]
    if occupied:
        raise RuntimeError(f"Refusing to overwrite T4 result files: {occupied}")
    pairs, pair_audit = read_pairs()
    decisions_by_model = decisions()
    config6 = MADOS_CONFIG / CELLS["full_scene_6ch"][1]
    mados6 = config_paths(config6, "test")
    selected, selection_audit = select_primary(pairs, mados6)
    chosen_keys = {(row["mados_id"], row["marida_patch"]) for row in selected}
    candidate_rows = []
    for pair in pairs:
        candidate_rows.append({key: pair[key] for key in ("mados_id", "marida_patch", "marida_split", "dy", "dx",
                               "match_score", "same_acq", "gt_centers_covered")}
                              | {"is_primary": (pair["mados_id"], pair["marida_patch"]) in chosen_keys})
    primary_rows = []
    for pair in selected:
        primary_rows.append({key: pair[key] for key in ("mados_id", "marida_patch", "marida_split", "dy", "dx",
                         "match_score", "same_acq", "gt_centers_covered", "mados_gt_all")})
    per_pair = []
    provenance = {"format": "mados-marida-t4-v1", "analysis_plan_sha256": plan_hash,
                  "script_sha256": sha256(Path(__file__)), "input_sha256": {str(path.relative_to(ROOT)): sha256(path)
                  for path in (MATCHES, RADIOMETRY, T2_SUMMARY, DECISIONS, INVENTORY)},
                  "t2_stop_signal": t2_summary["stop_signal"], "pair_audit": pair_audit,
                  "selection_audit": selection_audit, "models": {}}
    for cell, (marida_name, mados_name) in CELLS.items():
        marida_config = MARIDA_CONFIG / marida_name
        mados_config = MADOS_CONFIG / mados_name
        marida_paths = {split: config_paths(marida_config, split) for split in ("train", "val", "test")}
        mados_paths = config_paths(mados_config, "test")
        marida_root = Path(str(yaml.safe_load(marida_config.read_text(encoding="utf-8"))["path"]))
        mados_root = Path(str(yaml.safe_load(mados_config.read_text(encoding="utf-8"))["path"]))
        channels = 6 if cell == "full_scene_6ch" else 7
        config_hashes = {"marida_config": sha256(marida_config), "mados_config": sha256(mados_config),
                         "marida_stats": sha256(marida_root / f"channel_stats_{channels}ch.json"),
                         "mados_stats": sha256(mados_root / f"channel_stats_{channels}ch.json")}
        if config_hashes["marida_stats"] != config_hashes["mados_stats"]:
            raise RuntimeError(f"MADOS does not use frozen MARIDA statistics for {cell}")
        for seed in SEEDS:
            decision = decisions_by_model[(cell, seed)]
            if sha256(marida_config) != decision["dataset_config_sha256"]:
                raise RuntimeError(f"Locked MARIDA config changed: {marida_config}")
            run_dir = Path(str(decision["run_dir"]))
            val, val_hash = old_ledger(run_dir, "val", decision)
            test, test_hash = old_ledger(run_dir, "test", decision)
            fresh, cache_path = infer_missing(cell, seed, decision,
                          {"marida_train": marida_paths["train"], "mados": mados_paths,
                           "marida_config": marida_config, "mados_config": mados_config},
                          selected, args.device, config_hashes)
            provenance["models"][f"{cell}_s{seed}"] = {"weight_sha256": decision["weights_sha256"],
                "balanced_threshold": float(decision["balanced_threshold"]), "val_ledger_sha256": val_hash,
                "test_ledger_sha256": test_hash, "new_ledger": str(cache_path.relative_to(ROOT)),
                "new_ledger_sha256": sha256(cache_path), "config_and_stats_sha256": config_hashes}
            for pair in selected:
                marida_id, mados_id = str(pair["marida_patch"]), str(pair["mados_id"])
                split = str(pair["marida_split"])
                if split == "train":
                    mar = fresh["marida_train"].get(marida_id)
                elif split == "val":
                    mar = val.get(marida_id)
                else:
                    mar = test.get(marida_id)
                mad = fresh["mados"].get(mados_id)
                if mar is None or mad is None:
                    raise RuntimeError(f"Missing ledger entry for {mados_id}, {marida_id}")
                if split in {"val", "test"}:
                    if marida_id not in marida_paths[split]:
                        raise RuntimeError(f"Matched patch absent from current {split} manifest: {marida_id}")
                    verify_gt(mar, marida_paths[split][marida_id], 256)
                for threshold_name, threshold in (("min_conf", MIN_CONF),
                                                  ("balanced", float(decision["balanced_threshold"]))):
                    for record in score_pair(pair, mar, mad, threshold):
                        per_pair.append({"cell": cell, "seed": seed, "mados_id": mados_id,
                            "marida_patch": marida_id, "marida_split": split,
                            "split_group": "train" if split == "train" else "val_test",
                            "same_acq": "true" if pair["same_acq"] is True else "false" if pair["same_acq"] is False else "undetermined",
                            "threshold_name": threshold_name, "threshold": threshold, **record})
    summary = {}
    for row in per_pair:
        key = tuple(row[name] for name in ("cell", "seed", "split_group", "same_acq", "threshold_name", "grid"))
        if key not in summary:
            summary[key] = {name: row[name] for name in ("cell", "seed", "split_group", "same_acq",
                                                       "threshold_name", "threshold", "grid")}
            summary[key].update(pairs=0, tp=0, fp=0, fn=0, hit_gt=0, gt=0, pred=0)
        for name in ("tp", "fp", "fn", "hit_gt", "gt", "pred"):
            summary[key][name] += int(row[name])
        summary[key]["pairs"] += 1
    summary_rows = list(summary.values())
    summary_rows.sort(key=lambda row: (row["cell"], row["seed"], row["split_group"], row["same_acq"],
                                       row["threshold_name"], row["grid"]))
    write_csv(OUT / "t4_candidate_pairs.csv", candidate_rows, list(candidate_rows[0]))
    write_csv(OUT / "t4_primary_pairs.csv", primary_rows, list(primary_rows[0]))
    write_csv(OUT / "t4_pair_counts.csv", per_pair, list(per_pair[0]))
    write_csv(OUT / "t4_summary.csv", summary_rows, list(summary_rows[0]))
    metadata_path = OUT / "t4_provenance.json"
    metadata_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pair_audit": pair_audit, "selection_audit": selection_audit,
                      "summary_rows": len(summary_rows), "same_acq_true_pairs": sum(row["same_acq"] is True for row in selected),
                      "outputs": ["t4_candidate_pairs.csv", "t4_primary_pairs.csv", "t4_pair_counts.csv",
                                  "t4_summary.csv", "t4_provenance.json"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
