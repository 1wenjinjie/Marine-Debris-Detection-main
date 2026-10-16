"""
cv_pipeline.py
Balanced 9-fold leave-scene-out cross-validation over all 63 MARIDA scenes.

For each fold:
  test  = fold k scenes
  val   = fold (k+1)%9 scenes
  train = the other 7 folds
Image .npy and labels are reused in place via ABSOLUTE manifest paths (no copy);
channel p01/p99 statistics are recomputed from that fold's train split only.
Trains C6 and C7 with 3 paired seeds, collects out-of-fold (OOF) test predictions,
merges them into one 63-scene ledger, and reruns clustered bootstrap, paired
scene-swap permutation, leave-one-scene jackknife, and margin-based reading.

Modes:
  --smoke   one fold, one seed, 1 epoch, capped patches (writes under *_smoke)
  default   full 9 folds x 3 seeds x 2 channels = 54 trainings
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\OceanEdge_Group\Marine-Debris-Detection-main")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "mpb"))

import run_experiments as rex  # noqa: E402
from prepare_protocol import _compute_stats  # noqa: E402
import audit_toolkit as at  # noqa: E402

SRC_DS = ROOT / "data" / "processed" / "marida_mpb_v1" / "datasets"
AUDIT_DIR = ROOT / "data" / "processed" / "marida_mpb_v1" / "audit"
CV_ROOT = ROOT / "data" / "processed" / "marida_cv_v1"
RUNS_CV = ROOT / "runs_cv"
OOF_DIR = RUNS_CV / "oof"

# key -> (channels, source variant dir, image folder name)
CHDEF = {
    "c6": (6, "scene_6ch_min4", "images"),
    "c7": (7, "scene_7ch_ndwi_min4", "images_7ch"),
}
FULL_SEEDS = (0, 1, 2)
MARGIN = 0.05
NAMES = {0: "Marine_Debris", 1: "Algae_Organic", 2: "Ship"}
MATCH_IOU = 0.5


# --------------------------------------------------------------------------
# Scene / patch inventory
# --------------------------------------------------------------------------
def load_inventory():
    stem_scene = {}
    scene_stems = defaultdict(list)
    scene_min4 = defaultdict(int)
    with (AUDIT_DIR / "patch_inventory.csv").open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            stem_scene[row["stem"]] = row["scene"]
            scene_stems[row["scene"]].append(row["stem"])
            scene_min4[row["scene"]] += int(row["md_instances_min4"])
    return stem_scene, scene_stems, dict(scene_min4)


STEM_SCENE, SCENE_STEMS, SCENE_MIN4 = load_inventory()
ALL_SCENES = sorted(SCENE_STEMS)
FOLDS = at.grouped_scene_folds({s: SCENE_MIN4[s] for s in ALL_SCENES}, k=9, seed=20261012)


def source_npy_map(variant: str) -> dict[str, Path]:
    m = {}
    for split in ("train", "val", "test"):
        mf = SRC_DS / variant / f"{split}.txt"
        for line in mf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = Path(line.strip())
                m[p.stem] = p
    return m


SRC_NPY = {k: source_npy_map(v[1]) for k, v in CHDEF.items()}


# --------------------------------------------------------------------------
# Matching (full-confidence tp sequence for AP)
# --------------------------------------------------------------------------
def box_iou_one_vs_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    ab = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    aa = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return inter / np.maximum(aa + ab - inter, 1e-12)


def match_patch_tp(preds, gts):
    tp = np.zeros(len(preds))
    used = np.zeros(len(gts), bool)
    for i in np.argsort(-preds[:, 4], kind="stable"):
        avail = np.where(~used)[0]
        if not len(avail):
            break
        ious = box_iou_one_vs_many(preds[i, :4], gts[avail])
        j = int(np.argmax(ious))
        if ious[j] >= MATCH_IOU:
            tp[i] = 1.0
            used[avail[j]] = True
    return tp


def entries_to_scene_ledger(entries):
    led = {}
    for e in entries:
        scene = e["scene"]
        preds = np.asarray(e["predictions"], float).reshape(-1, 6)
        preds = preds[preds[:, 5] == 0.0]
        gts = np.asarray(e["ground_truth"], float).reshape(-1, 4)
        tp = match_patch_tp(preds, gts)
        s0, t0, n0 = led.get(scene, (np.array([]), np.array([]), 0))
        led[scene] = (
            np.concatenate([s0, preds[:, 4]]) if len(s0) else preds[:, 4].copy(),
            np.concatenate([t0, tp]) if len(t0) else tp.copy(),
            n0 + len(gts),
        )
    return led


# --------------------------------------------------------------------------
# Per-fold dataset preparation
# --------------------------------------------------------------------------
def fold_scene_sets(k):
    test = set(FOLDS[k])
    val = set(FOLDS[(k + 1) % 9])
    train = set(ALL_SCENES) - test - val
    return train, val, test


def cap_stems(stems, per_scene_cap):
    if per_scene_cap is None:
        return stems
    by_scene = defaultdict(list)
    for st in stems:
        by_scene[STEM_SCENE[st]].append(st)
    out = []
    for sc, lst in by_scene.items():
        out.extend(sorted(lst)[:per_scene_cap])
    return sorted(out)


def prepare_fold(k, chkey, smoke=False):
    channels, variant, img_folder = CHDEF[chkey]
    npy_map = SRC_NPY[chkey]
    train_sc, val_sc, test_sc = fold_scene_sets(k)

    train_stems = cap_stems([st for sc in train_sc for st in SCENE_STEMS[sc]],
                            2 if smoke else None)
    val_stems = cap_stems([st for sc in val_sc for st in SCENE_STEMS[sc]],
                          2 if smoke else None)
    test_stems = cap_stems([st for sc in test_sc for st in SCENE_STEMS[sc]],
                           None)

    tag = f"fold{k}_{chkey}" + ("_smoke" if smoke else "")
    ddir = CV_ROOT / tag
    ddir.mkdir(parents=True, exist_ok=True)

    for split, stems in (("train", train_stems), ("val", val_stems), ("test", test_stems)):
        missing = [st for st in stems if st not in npy_map]
        if missing:
            raise RuntimeError(f"{tag}/{split} missing source npy: {missing[:3]}")
        (ddir / f"{split}.txt").write_text(
            "\n".join(str(npy_map[st]) for st in stems), encoding="utf-8")

    stats = _compute_stats([npy_map[st] for st in train_stems], channels)
    (ddir / f"channel_stats_{channels}ch.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    (ddir / "variant_metadata.json").write_text(json.dumps({
        "name": tag,
        "split_kind": "scene_cv",
        "source_kind": "6ch" if channels == 6 else "7ch_ndwi",
        "channels": channels,
        "train_min_pixels_marine_debris": 4,
        "val_test_min_pixels_marine_debris": 4,
        "min_pixels_other_classes": 4,
        "fold": k,
        "val_fold": (k + 1) % 9,
    }, indent=2), encoding="utf-8")

    yaml_text = (
        f"path: {ddir}\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
        f"channels: {channels}\nnc: 3\nnames:\n"
        + "\n".join(f"  {i}: {n}" for i, n in NAMES.items())
        + "\n"
    )
    (ddir / "cv.yaml").write_text(yaml_text, encoding="utf-8")
    return ddir, {"train": len(train_stems), "val": len(val_stems), "test": len(test_stems)}


# --------------------------------------------------------------------------
# Train + collect OOF
# --------------------------------------------------------------------------
def run_fold(k, chkey, seed, device, smoke=False):
    channels, _, _ = CHDEF[chkey]
    ddir, counts = prepare_fold(k, chkey, smoke)
    tag = f"fold{k}_{chkey}" + ("_smoke" if smoke else "")
    run_name = f"{tag}_s{seed}"
    run_dir = (RUNS_CV / ("smoke" if smoke else "")) / run_name
    oof_path = OOF_DIR / f"{run_name}_oof.json"

    if oof_path.exists() and (run_dir / "weights" / "best.pt").exists() and not smoke:
        print(f"[skip] {run_name} OOF exists")
        return oof_path

    hyper = {**rex.LOCKED_HYPERPARAMS}
    if smoke:
        hyper.update(epochs=1, patience=1, workers=0, plots=False)
    init = rex._ensure_initialization()[channels]
    overrides = {
        **hyper,
        "model": str(init),
        "data": str(ddir / "cv.yaml"),
        "project": str(RUNS_CV / ("smoke" if smoke else "")),
        "name": run_name,
        "seed": seed,
        "device": device,
        "exist_ok": True,
    }
    print(f"[train] {run_name} counts={counts}")
    trainer = rex.MPBTrainer(overrides=overrides)
    trainer.train()
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)

    val_entries = rex.collect_predictions(best, ddir / "cv.yaml", "val",
                                          imgsz=int(hyper["imgsz"]), device=device)
    rex._write_prediction_ledger(run_dir / "val_predictions.json", val_entries)

    test_entries = rex.collect_predictions(best, ddir / "cv.yaml", "test",
                                           imgsz=int(hyper["imgsz"]), device=device,
                                           allow_test=True)
    rex._write_prediction_ledger(oof_path, test_entries)
    print(f"[oof] {oof_path.name} entries={len(test_entries)}")
    return oof_path


# --------------------------------------------------------------------------
# Merge OOF and run 63-cluster inference
# --------------------------------------------------------------------------
def load_validated_oof(seeds, smoke=False):
    """Validate the full paired inventory before writing any merged outputs."""
    seeds = tuple(seeds)
    if not smoke and seeds != FULL_SEEDS:
        raise RuntimeError(f"Formal OOF requires seeds {FULL_SEEDS}, received {seeds}")
    folds = tuple(range(1) if smoke else range(9))
    paths = {
        (k, ch, seed): OOF_DIR / (
            f"fold{k}_{ch}" + ("_smoke" if smoke else "") + f"_s{seed}_oof.json")
        for k in folds for ch in CHDEF for seed in seeds
    }
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"Incomplete OOF: {len(paths) - len(missing)}/{len(paths)} required files; "
            f"missing {', '.join(missing)}. No merged or final report written.")

    if not smoke:
        fold_scenes = [sc for group in FOLDS for sc in group]
        if len(fold_scenes) != 63 or len(set(fold_scenes)) != 63 or set(fold_scenes) != set(ALL_SCENES):
            raise RuntimeError("Fold inventory must cover exactly 63 scenes once")
        if len(STEM_SCENE) != 1381 or sum(SCENE_MIN4.values()) != 212:
            raise RuntimeError("Source inventory must contain 1381 patches and 212 MIN4 debris GT")

    entries_by_run = {}
    fold_details = []
    reference_seed = seeds[0]
    for k in folds:
        train_sc, val_sc, test_sc = fold_scene_sets(k)
        expected_stems = {st for sc in test_sc for st in SCENE_STEMS[sc]}
        for ch in CHDEF:
            for seed in seeds:
                key = (k, ch, seed)
                entries = rex._read_prediction_ledger(paths[key])
                by_stem = {e["stem"]: e for e in entries}
                if len(by_stem) != len(entries) or set(by_stem) != expected_stems:
                    raise RuntimeError(f"{paths[key].name}: duplicate, missing or extra test patches")
                for st, e in by_stem.items():
                    if e["scene"] != STEM_SCENE[st]:
                        raise RuntimeError(f"{paths[key].name}: incorrect scene for {st}")
                for sc in test_sc:
                    count = sum(len(e["ground_truth"]) for e in entries if e["scene"] == sc)
                    if count != SCENE_MIN4[sc]:
                        raise RuntimeError(f"{paths[key].name}: GT count for {sc}: {count} != {SCENE_MIN4[sc]}")
                if key != (k, "c6", reference_seed):
                    reference = {e["stem"]: e for e in entries_by_run[(k, "c6", reference_seed)]}
                    for st, e in by_stem.items():
                        if not np.array_equal(e["ground_truth"], reference[st]["ground_truth"]):
                            raise RuntimeError(f"{paths[key].name}: GT differs across channels/seeds for {st}")
                entries_by_run[key] = entries
        fold_details.append({
            "fold": k,
            "splits": {
                name: {"scenes": sorted(scene_set), "n_scenes": len(scene_set),
                       "n_patches": sum(len(SCENE_STEMS[sc]) for sc in scene_set),
                       "n_gt": sum(SCENE_MIN4[sc] for sc in scene_set)}
                for name, scene_set in (("train", train_sc), ("val", val_sc), ("test", test_sc))
            },
        })
    return entries_by_run, fold_details


def merge_and_analyze(seeds, smoke=False):
    seeds = tuple(seeds)
    entries_by_run, fold_details = load_validated_oof(seeds, smoke)
    report = {"smoke": smoke, "channels": {}, "contrast_c7_minus_c6": {},
              "seeds": list(seeds), "fold_inventory": fold_details,
              "per_fold_per_seed": []}
    led_by_ch_seed = {ch: [] for ch in CHDEF}

    for detail in fold_details:
        k = detail["fold"]
        scenes = detail["splits"]["test"]["scenes"]
        for seed in seeds:
            aps = {ch: at.pooled_ap(entries_to_scene_ledger(entries_by_run[(k, ch, seed)]), scenes)
                   for ch in CHDEF}
            report["per_fold_per_seed"].append({
                "fold": k, "seed": seed, "c6_AP": aps["c6"], "c7_AP": aps["c7"],
                "c7_minus_c6": aps["c7"] - aps["c6"],
            })

    for chkey in CHDEF:
        merged_per_seed = []
        for seed in seeds:
            merged = []
            folds = range(1) if smoke else range(9)
            for k in folds:
                merged.extend(entries_by_run[(k, chkey, seed)])
            if not smoke:
                mp = RUNS_CV / f"cv_merged_{chkey}_s{seed}.json"
                rex._write_prediction_ledger(mp, merged)
            ledger = entries_to_scene_ledger(merged)
            led_by_ch_seed[chkey].append(ledger)
            merged_per_seed.append(ledger)
        scenes = sorted(merged_per_seed[0])
        aps = [at.pooled_ap(lg, scenes) for lg in merged_per_seed]
        report["channels"][chkey] = {
            "n_scenes": len(scenes),
            "per_seed_AP": aps,
            "mean_AP": round(float(np.mean(aps)), 4),
        }

    if smoke:
        return report

    C6 = led_by_ch_seed["c6"]
    C7 = led_by_ch_seed["c7"]
    scenes = sorted(C6[0])
    report["integrity"] = {
        "status": "passed", "n_required_formal_runs": 54,
        "n_oof_scenes_per_channel_seed": len(scenes),
        "n_oof_patches_per_channel_seed": sum(d["splits"]["test"]["n_patches"] for d in fold_details),
        "n_oof_gt_per_channel_seed": sum(d["splits"]["test"]["n_gt"] for d in fold_details),
        "each_scene_and_patch_tested_once": True, "same_GT_across_channels_and_seeds": True,
        "smoke_files_excluded": True,
    }
    report["inference_scope"] = {
        "statistic": "Mean across paired seeds of globally pooled OOF AP(C7) - AP(C6); not mean fold AP",
        "conditioning": "Scene resampling and swaps condition on the 54 fitted models and this fold allocation. Overlapping training sets induce dependence between folds; this is not an unconditional estimate of training or split uncertainty.",
        "permutation_null": "Model prediction ledgers are exchangeable within each acquisition scene; the same scene swap applies to all paired seeds.",
        "margin": MARGIN, "margin_status": "Post hoc supplementary reading, not a preregistered equivalence margin",
        "ci_vs_margin": "Descriptive reading of the 95% bootstrap interval, not a formal TOST equivalence test",
        "pooled_score_limitation": "OOF pooling ranks confidence scores across separately fitted fold models; fold-specific score calibration can affect pooled AP.",
    }
    report["per_seed_pooled_contrast"] = [
        {"seed": seed, "c6_AP": at.pooled_ap(c6, scenes), "c7_AP": at.pooled_ap(c7, scenes),
         "c7_minus_c6": at.pooled_ap(c7, scenes) - at.pooled_ap(c6, scenes)}
        for seed, c6, c7 in zip(seeds, C6, C7)
    ]
    pt, (lo, hi), _ = at.scene_bootstrap(C7, C6, B=10000, seed=20261012)
    obs, p_perm, _ = at.paired_scene_swap(C7, C6, n_mc=20000, seed=20261012)
    full_jk, jk = at.scene_jackknife(C7, C6)
    jk_sorted = sorted(jk.items(), key=lambda kv: kv[1])
    report["contrast_c7_minus_c6"] = {
        "n_scenes": len(scenes),
        "point": round(pt, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "permutation_p_two_sided": round(p_perm, 4),
        "bootstrap_replicates": 10000,
        "permutation_mc_draws": 20000,
        "permutation_p_correction": "(extreme + 1) / (n_mc + 1)",
        "jackknife_full": round(full_jk, 4),
        "jackknife_min": {"scene": jk_sorted[0][0], "diff": round(jk_sorted[0][1], 4)},
        "jackknife_max": {"scene": jk_sorted[-1][0], "diff": round(jk_sorted[-1][1], 4),
                          "n_gt_in_scene": int(C6[0][jk_sorted[-1][0]][2])},
        f"ci_vs_margin_{MARGIN}": at.ci_vs_margin(lo, hi, MARGIN),
        "jackknife_all": {sc: round(v, 4) for sc, v in sorted(jk.items())},
    }
    return report


PILOT_JOBS = [(2, "c6", 0), (2, "c7", 0), (2, "c6", 1)]


def fold_split_counts(k):
    tr, va, te = fold_scene_sets(k)
    return (sum(len(SCENE_STEMS[s]) for s in tr),
            sum(len(SCENE_STEMS[s]) for s in va),
            sum(len(SCENE_STEMS[s]) for s in te))


def read_run_timing(run_dir: Path):
    p = run_dir / "results.csv"
    with p.open(encoding="utf-8", newline="") as fh:
        rows = [{k.strip(): (v.strip() if v else v) for k, v in r.items()}
                for r in csv.DictReader(fh)]
    times = [float(r["time"]) for r in rows]
    return {"n_epochs": len(times),
            "total_s": round(times[-1], 1),
            "per_epoch_s": round(times[-1] / len(times), 2)}


def run_pilot(device: str):
    rows = []
    for k, ch, seed in PILOT_JOBS:
        run_fold(k, ch, seed, device, smoke=False)
        run_dir = RUNS_CV / f"fold{k}_{ch}_s{seed}"
        t = read_run_timing(run_dir)
        tr, va, te = fold_split_counts(k)
        t.update({"fold": k, "channel": ch, "seed": seed,
                  "train_patches": tr, "val_patches": va, "test_patches": te})
        rows.append(t)

    # Full run: 9 folds x 2 channels x 3 seeds = 54 jobs.
    n_ch, n_seed, n_folds = 2, len(FULL_SEEDS), 9
    train_patches_all = 9667 * n_ch * n_seed
    val_patches_all = 1381 * n_ch * n_seed
    oof_patches_all = 1381 * n_ch * n_seed
    ref_load = rows[0]["train_patches"] + rows[0]["val_patches"]
    mean_total_s = float(np.mean([r["total_s"] for r in rows]))
    scale = (train_patches_all + val_patches_all) / ref_load
    train_wall_h = mean_total_s * scale / 3600.0
    # extra one-time ledger passes (val-ledger collect + OOF collect), ~10 ms/patch
    extra_h = (val_patches_all + oof_patches_all) * 0.010 / 3600.0

    summary = {
        "pilot_jobs": rows,
        "mean_single_training_min": round(mean_total_s / 60.0, 2),
        "reference_load_train_plus_val": ref_load,
        "projected_total_train_wall_hours": round(train_wall_h, 1),
        "projected_extra_ledger_hours": round(extra_h, 2),
        "projected_total_hours": round(train_wall_h + extra_h, 1),
        "n_jobs_full": n_folds * n_ch * n_seed,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    RUNS_CV.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        run_fold(0, "c6", 0, args.device, smoke=True)
        run_fold(0, "c7", 0, args.device, smoke=True)
        report = merge_and_analyze((0,), smoke=True)
        print("SMOKE OK:", json.dumps(report, indent=2, ensure_ascii=False))
        return

    if args.pilot:
        summary = run_pilot(args.device)
        rex._json_dump(RUNS_CV / "pilot_timing.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if not args.analyze_only:
        for k in range(9):
            for chkey in ("c6", "c7"):
                for seed in FULL_SEEDS:
                    run_fold(k, chkey, seed, args.device, smoke=False)

    report = merge_and_analyze(FULL_SEEDS, smoke=False)
    out = RUNS_CV / "cv_stat_results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("written:", out)


if __name__ == "__main__":
    main()
