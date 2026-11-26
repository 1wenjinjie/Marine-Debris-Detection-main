"""Phase 3B: Zero-shot external validation of frozen D4 checkpoints on MADOS and PLP2021.

Completely isolated from historical runs_mpb/external_*:
  - Writes MADOS results to runs_mpb_d4/external_mados/ (full + T6 overlap-filtered + val-locked op)
  - Writes PLP2021 results to runs_mpb_d4/external_plp2021/ (both 6ch and 7ch, val-locked + conf>=0.001)
No MADOS or PLP labels are ever used for model, normalization, or threshold selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "mpb"))

import run_experiments as rex  # noqa: E402
import run_plp2021_external_validation as plp  # noqa: E402
from scripts.v2_evaluate import evaluate  # noqa: E402

MPB_D4 = ROOT / "runs_mpb_d4"
MPB_FULL = MPB_D4 / "full"
OUT_MADOS = MPB_D4 / "external_mados"
OUT_PLP = MPB_D4 / "external_plp2021"

MADOS_CFG = {
    "full_scene_6ch": (ROOT / "configs/datasets/mpb/mados_external_scene_6ch_min4.yaml", 6),
    "full_scene_ndwi7": (ROOT / "configs/datasets/mpb/mados_external_scene_7ch_ndwi_min4.yaml", 7),
}
MARIDA_STATS = {
    "full_scene_6ch": ROOT / "data/processed/marida_mpb_v1/datasets/scene_6ch_min4/channel_stats_6ch.json",
    "full_scene_ndwi7": ROOT / "data/processed/marida_mpb_v1/datasets/scene_7ch_ndwi_min4/channel_stats_7ch.json",
}
DUP_MAP = ROOT / "runs_mpb/mados_overlap/v2/dup_map_full.csv"
T6_SUMMARY = ROOT / "runs_mpb/mados_overlap/v2/t6_summary.json"
SEEDS = (0, 1, 2)
IMAGE_SIZE = 512


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_d4_locked_runs(cell: str) -> list[dict]:
    rows = []
    for s in SEEDS:
        run_dir = MPB_FULL / f"{cell}_s{s}"
        w = run_dir / "weights" / "best.pt"
        ev = json.loads((run_dir / "eval_test_locked.json").read_text(encoding="utf-8"))
        if not w.is_file():
            raise FileNotFoundError(w)
        rows.append({
            "cell": cell,
            "seed": s,
            "weights": str(w),
            "weights_sha256": sha256_file(w),
            "balanced_threshold": float(ev["val_locked_balanced_threshold"]),
        })
    return rows


def load_t6_exclusion_sets() -> dict[str, set[str]]:
    expected = json.loads(T6_SUMMARY.read_text(encoding="utf-8"))["dup_map_sha256"]
    if sha256_file(DUP_MAP) != expected:
        raise RuntimeError("dup_map_full.csv hash mismatch")
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    with DUP_MAP.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["mados_split"] == "test" and row["same_acq"] == "True":
                by_split[row["marida_split"]].add(row["mados_id"])
    return {
        "full": set(),
        "t6_test_vs_marida_train": set(by_split["train"]),
        "t6_test_vs_marida_trainval": by_split["train"] | by_split["val"],
        "t6_test_vs_marida_any": by_split["train"] | by_split["val"] | by_split["test"],
    }


def make_filtered_mados_yaml(base_yaml: Path, variant: str, exclude_ids: set[str], out_dir: Path) -> tuple[Path, dict]:
    cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    ds_root = Path(cfg["path"])
    raw_lines = [ln.strip() for ln in (ds_root / cfg["test"]).read_text(encoding="utf-8").splitlines() if ln.strip()]
    kept = []
    n_md_gt = 0
    n_pos_tiles = 0
    for ln in raw_lines:
        p = Path(ln)
        if p.stem in exclude_ids:
            continue
        kept.append(ln)
        lbl = p.parent.parent / "labels" / f"{p.stem}.txt"
        cnt = sum(1 for row in lbl.read_text(encoding="utf-8").splitlines() if row.strip() and row.split()[0] == "0")
        n_md_gt += cnt
        if cnt > 0:
            n_pos_tiles += 1
    vdir = out_dir / "filtered_configs" / f"{variant}_{cfg['channels']}ch"
    vdir.mkdir(parents=True, exist_ok=True)
    test_txt = vdir / "test.txt"
    test_txt.write_text("\n".join(kept) + "\n", encoding="utf-8")
    new_cfg = {**cfg, "test": str(test_txt.resolve())}
    y_path = vdir / "dataset.yaml"
    y_path.write_text(yaml.safe_dump(new_cfg, sort_keys=False), encoding="utf-8")
    meta = {
        "excluded_crops": len(raw_lines) - len(kept),
        "retained_crops": len(kept),
        "retained_md_gt_min4": n_md_gt,
        "retained_md_pos_tiles": n_pos_tiles,
    }
    return y_path, meta


def run_mados(device: str) -> dict:
    print("\n" + "=" * 72, flush=True)
    print("[Phase 3B-1] MADOS Zero-Shot External Evaluation (Full + T6 Filtered + Val-Locked OP) ...", flush=True)
    print("=" * 72, flush=True)
    if OUT_MADOS.exists():
        shutil.rmtree(OUT_MADOS)
    OUT_MADOS.mkdir(parents=True, exist_ok=True)

    excl_sets = load_t6_exclusion_sets()
    variants_out = {}

    # Collect per-seed raw prediction ledgers on full MADOS once (for exact 101-pt AP + val-locked F1 across all subsets)
    full_ledgers: dict[tuple[str, int], list[dict]] = {}
    locked_info: dict[str, list[dict]] = {}
    for cell, (base_yaml, ch) in MADOS_CFG.items():
        locked_info[cell] = load_d4_locked_runs(cell)
        for row in locked_info[cell]:
            s = row["seed"]
            print(f"  [ledger] MADOS {cell} s{s} ...", flush=True)
            ents = rex.collect_predictions(Path(row["weights"]), base_yaml, "test", imgsz=IMAGE_SIZE, device=device, allow_test=True)
            full_ledgers[(cell, s)] = ents
            rex._write_prediction_ledger(OUT_MADOS / "ledgers" / f"{cell}_s{s}_mados_test.json", ents)

    for vname, excl in excl_sets.items():
        print(f"\n  --- MADOS variant: {vname} (excluding {len(excl)} crops) ---", flush=True)
        v_cells = {}
        v_meta = None
        for cell, (base_yaml, ch) in MADOS_CFG.items():
            if excl:
                eval_yaml, meta = make_filtered_mados_yaml(base_yaml, vname, excl, OUT_MADOS)
            else:
                eval_yaml, meta = base_yaml, {
                    "excluded_crops": 0, "retained_crops": 728,
                    "retained_md_gt_min4": 90, "retained_md_pos_tiles": 26,
                }
            v_meta = meta
            per_seed = []
            for row in locked_info[cell]:
                s = row["seed"]
                thr = row["balanced_threshold"]
                # Ultralytics val (exact parity with run_mados_external_validation.py)
                u_metrics = evaluate(
                    weights=row["weights"],
                    data_yaml=str(eval_yaml),
                    channels=ch,
                    split="test",
                    imgsz=IMAGE_SIZE,
                    batch=8,
                    device=device,
                    conf=0.001,
                    iou=0.6,
                    workers=0,
                    project=str(OUT_MADOS / "ultralytics_eval" / vname),
                    name=f"{cell}_s{s}",
                    allow_test=True,
                )
                # Ledger AP50 + strictly MARIDA val-locked operating point on retained crops
                sub_ents = [e for e in full_ledgers[(cell, s)] if e["stem"] not in excl]
                led_ap50 = float(rex.ap50(sub_ents))
                op_locked = rex.operating_metrics(sub_ents, thr)
                md_u = u_metrics["per_class"]["Marine_Debris"]
                per_seed.append({
                    "seed": s,
                    "weights": row["weights"],
                    "weights_sha256": row["weights_sha256"],
                    "val_locked_threshold": thr,
                    "ultralytics_mAP50": float(u_metrics["mAP50"]),
                    "ultralytics_MD_AP50": float(md_u["AP50"]),
                    "ultralytics_MD_AP50_95": float(md_u["AP50_95"]),
                    "ledger_MD_AP50": led_ap50,
                    "val_locked_op": op_locked,
                })
                print(
                    f"    {vname} | {cell} s{s}: Ultra MD_AP50={md_u['AP50']:.4f}, "
                    f"Ledger AP50={led_ap50:.4f}, val_thr={thr} -> "
                    f"P={op_locked['precision']:.3f} R={op_locked['recall']:.3f} F1={op_locked['f1']:.3f} "
                    f"(TP={op_locked['tp']} FP={op_locked['fp']})",
                    flush=True,
                )
            v_cells[cell] = {
                "mean_ultralytics_mAP50": round(float(np.mean([r["ultralytics_mAP50"] for r in per_seed])), 4),
                "mean_ultralytics_MD_AP50": round(float(np.mean([r["ultralytics_MD_AP50"] for r in per_seed])), 4),
                "std_ultralytics_MD_AP50": round(float(np.std([r["ultralytics_MD_AP50"] for r in per_seed], ddof=1)), 4),
                "mean_ledger_MD_AP50": round(float(np.mean([r["ledger_MD_AP50"] for r in per_seed])), 4),
                "mean_val_locked_F1": round(float(np.mean([r["val_locked_op"]["f1"] for r in per_seed])), 4),
                "mean_val_locked_P": round(float(np.mean([r["val_locked_op"]["precision"] for r in per_seed])), 4),
                "mean_val_locked_R": round(float(np.mean([r["val_locked_op"]["recall"] for r in per_seed])), 4),
                "mean_val_locked_FP_per_100": round(float(np.mean([r["val_locked_op"]["false_alarms_per_100_patches"] for r in per_seed])), 2),
                "per_seed": per_seed,
            }
        variants_out[vname] = {"counts": v_meta, "cells": v_cells}

    summary = {
        "generated_utc": utc_now(),
        "augmentation": "RandomD4",
        "normalization_rule": "MARIDA train-only p01/p99 imported without refitting",
        "threshold_rule": "Strictly MARIDA validation-locked balanced threshold (MADOS never used)",
        "variants": variants_out,
    }
    write_json(OUT_MADOS / "summary.json", summary)
    print(f"[done] MADOS -> {OUT_MADOS / 'summary.json'}", flush=True)
    return summary


def preprocess_plp_array(array: np.ndarray, low: np.ndarray, high: np.ndarray, imgsz: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    array = np.clip(array, low, high)
    range_scale = np.float32(255.0) / np.maximum(high - low, np.float32(1e-6))
    array = (array - low) * range_scale
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    h, w = array.shape[:2]
    scale = imgsz / max(h, w)
    rw, rh = int(round(w * scale)), int(round(h * scale))
    import cv2
    resized = cv2.resize(array, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pw, ph = imgsz - rw, imgsz - rh
    left, top = pw // 2, ph // 2
    padded = np.pad(resized, ((top, ph - top), (left, pw - left), (0, 0)), mode="constant", constant_values=114.0)
    return padded.astype(np.float32), (left, top, rw, rh)


def run_plp2021(device: str) -> dict:
    print("\n" + "=" * 72, flush=True)
    print("[Phase 3B-2] PLP2021 Controlled External Check (6ch + 7ch, val-locked + conf>=0.001) ...", flush=True)
    print("=" * 72, flush=True)
    if OUT_PLP.exists():
        shutil.rmtree(OUT_PLP)
    OUT_PLP.mkdir(parents=True, exist_ok=True)

    scene_root = ROOT / "data/PLP/PLP2021_native256/Sentinel-2"
    if not scene_root.exists():
        scene_root = ROOT / "data/PLP/PLP2021_native256"
    scenes = plp._load_scenes(scene_root)
    plp._assert_native_crop_geometry(scenes.values())
    targets = plp._load_targets()
    torch_device = torch.device("cpu") if device == "cpu" or not torch.cuda.is_available() else torch.device(f"cuda:{device}")

    cells_out = {}
    for cell, ch in (("full_scene_6ch", 6), ("full_scene_ndwi7", 7)):
        stats = json.loads(MARIDA_STATS[cell].read_text(encoding="utf-8"))
        low = np.asarray(stats["percentiles"]["p01"], dtype=np.float32).reshape(1, 1, ch)
        high = np.asarray(stats["percentiles"]["p99"], dtype=np.float32).reshape(1, 1, ch)
        runs = load_d4_locked_runs(cell)

        locked_rows = []
        conf001_rows = []
        for row in runs:
            s = row["seed"]
            thr = row["balanced_threshold"]
            yolo = YOLO(row["weights"])
            model = yolo.model.to(torch_device).eval()
            nc = int(model.yaml.get("nc", 3))

            preds_all = []
            for date, scene in sorted(scenes.items()):
                img7, transform = plp._load_scene_array(scene)
                arr = img7[..., :6] if ch == 6 else img7
                prep, geom = preprocess_plp_array(arr, low, high, IMAGE_SIZE)
                tensor = torch.from_numpy(prep.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0).to(torch_device)
                dets = rex._raw_predict(model, tensor, nc)
                deb = dets[dets[:, 5].astype(int) == 0]
                boxes = plp._boxes_to_native(deb[:, :4], geom, scene.width, scene.height)
                for box, det in zip(boxes, deb):
                    easting, northing = plp._world_center(transform, box)
                    preds_all.append({
                        "date": date,
                        "scene": scene.path.name,
                        "confidence": float(det[4]),
                        "class_id": int(det[5]),
                        "box_pixels_xyxy": [float(v) for v in box],
                        "center_easting": easting,
                        "center_northing": northing,
                    })

            preds_locked = [p for p in preds_all if p["confidence"] >= thr]
            preds_c001 = [p for p in preds_all if p["confidence"] >= 0.001]

            payload_locked = {"seed": s, "threshold": thr, "predictions": preds_locked}
            payload_c001 = {"seed": s, "threshold": 0.001, "predictions": preds_c001}
            write_json(OUT_PLP / f"{cell}_s{s}_locked.json", payload_locked)
            write_json(OUT_PLP / f"{cell}_s{s}_conf001.json", payload_c001)

            sum_locked = plp._summarize_seed(payload_locked, targets, scenes)
            sum_c001 = plp._summarize_seed(payload_c001, targets, scenes)
            locked_rows.append(sum_locked)
            conf001_rows.append(sum_c001)
            print(
                f"  PLP {cell} s{s} | locked(thr={thr}): HDPE_24m={sum_locked['hdpe_target_recovery_primary_24m']['detected']}/21, "
                f"wood={sum_locked['wood_target_marine_debris_assignments_24m']['detected']}/15, alerts={sum_locked['marine_debris_predictions']} | "
                f"conf>=0.001: HDPE_24m={sum_c001['hdpe_target_recovery_primary_24m']['detected']}/21, "
                f"wood={sum_c001['wood_target_marine_debris_assignments_24m']['detected']}/15, alerts={sum_c001['marine_debris_predictions']}",
                flush=True,
            )

        def agg(rows_list):
            return {
                "mean_hdpe_24m_detected": round(float(np.mean([r["hdpe_target_recovery_primary_24m"]["detected"] for r in rows_list])), 2),
                "mean_hdpe_24m_rate": round(float(np.mean([r["hdpe_target_recovery_primary_24m"]["rate"] for r in rows_list])), 4),
                "mean_hdpe_nominal_24m_rate": round(float(np.mean([r["hdpe_target_recovery_nominal_only_24m"]["rate"] for r in rows_list])), 4),
                "mean_wood_24m_detected": round(float(np.mean([r["wood_target_marine_debris_assignments_24m"]["detected"] for r in rows_list])), 2),
                "mean_total_alerts": round(float(np.mean([r["marine_debris_predictions"] for r in rows_list])), 2),
                "mean_unmatched_alerts_per_km2": round(float(np.mean([r["unmatched_alerts_per_km2"] for r in rows_list])), 4),
            }

        cells_out[cell] = {
            "validation_locked_aggregate": agg(locked_rows),
            "conf001_diagnostic_aggregate": agg(conf001_rows),
            "validation_locked_per_seed": locked_rows,
            "conf001_diagnostic_per_seed": conf001_rows,
        }

    summary = {
        "generated_utc": utc_now(),
        "augmentation": "RandomD4",
        "scene_count": len(scenes),
        "target_count": len(targets),
        "cells": cells_out,
    }
    write_json(OUT_PLP / "summary.json", summary)
    print(f"[done] PLP2021 -> {OUT_PLP / 'summary.json'}", flush=True)
    return summary


def main() -> int:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="0")
    args = p.parse_args()
    run_mados(args.device)
    run_plp2021(args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
