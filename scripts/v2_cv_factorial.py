"""
v2_cv_factorial.py
Scene-CV replication of the 2x2 audit: prior {ndwi, oracle} x exposure {clean, shared}.

Folds, validation folds, hyperparameters, initialisation, prediction collection and the
debris-class ledger AP are reused from scripts/mpb/cv_pipeline.py (cvp.FOLDS, seed 20261012).

Per fold k (test = FOLDS[k], val = FOLDS[(k+1)%9], base train = the other 7 folds):
  * every test scene with n>=2 patches is exposable: floor(n/2) patches become sibling
    exposure patches (debris-positive and other patches split separately, SHA-256 order);
    the remaining patches form the test panel. Single-patch scenes stay in the panel only.
  * removal set R: |R| = |S| patches drawn from base train, matched exactly on
    negative_category_min4 counts; the debris-positive part minimises the MIN4 instance gap
    over N_DRAWS seeded draws.
  * clean train = base; shared train = base - R + S; validation and test panel identical.
  * normalisation: one channel_stats_7ch.json per fold fitted on base - R (the exact
    clean/shared intersection) from NDWI arrays, prior channel fixed to [0, 1], shared by
    all four cells (same contract as the historical v1 audit).
  * oracle prior = per-patch normalised L2 distance transform of (MARIDA *_cl.tif > 0),
    identical to scripts/preprocessing/preprocess_generate_knowledge_channel.py.

Modes:
  --design    build and validate all 9 fold designs and oracle arrays, no training
  --smoke     fold 0, seed 0, four cells, 1 epoch, capped train/val patches
  (default)   train 9 folds x 4 cells x 3 seeds = 108 runs (resumable)
  --jobs 2    run two independent training tasks concurrently using spawn workers
  --seed 1 2  train only selected seeds (analysis still requires all 108 runs)
  --analyze   merge OOF ledgers and write runs_v2/_tmp_eval/cv_factorial
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"D:\OceanEdge_Group\Marine-Debris-Detection-main")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "mpb"))

from cv_factorial_scheduler import (  # noqa: E402
    build_tasks, file_lock, reject_other_schedulers, run_locked, schedule, write_consistent,
)

FACT_ROOT = ROOT / "data" / "processed" / "marida_cv_factorial_v1"
ORACLE_ROOT = FACT_ROOT / "oracle7"
DESIGN_JSON = FACT_ROOT / "design.json"
RUNS = ROOT / "runs_cv_factorial"
OOF_DIR = RUNS / "oof"
DESIGN_OUT = ROOT / "runs_v2" / "_tmp_eval" / "cv_factorial_design"
ANALYSIS_OUT = ROOT / "runs_v2" / "_tmp_eval" / "cv_factorial"
PLAN_PATH = ROOT / "runs_mpb" / "cv_factorial" / "analysis_plan.md"

SEEDS = (0, 1, 2)
CELLS = ("clean_ndwi", "clean_oracle", "shared_ndwi", "shared_oracle")
REMOVAL_SEED = 20261015
N_DRAWS = 2000
BOOT_SEED = 20261015
MARGIN = 0.05
POSITIVE = "marine_debris_positive"
CONTRASTS = {
    "scene_exposure_main_effect": {"clean_ndwi": -0.5, "clean_oracle": -0.5,
                                   "shared_ndwi": 0.5, "shared_oracle": 0.5},
    "oracle_prior_main_effect": {"clean_ndwi": -0.5, "clean_oracle": 0.5,
                                 "shared_ndwi": -0.5, "shared_oracle": 0.5},
    "interaction": {"clean_ndwi": 1.0, "clean_oracle": -1.0,
                    "shared_ndwi": -1.0, "shared_oracle": 1.0},
    "exposure_effect_ndwi": {"clean_ndwi": -1.0, "shared_ndwi": 1.0},
    "exposure_effect_oracle": {"clean_oracle": -1.0, "shared_oracle": 1.0},
    "oracle_effect_clean": {"clean_ndwi": -1.0, "clean_oracle": 1.0},
    "oracle_effect_shared": {"shared_ndwi": -1.0, "shared_oracle": 1.0},
}
PRIMARY_CONTRASTS = ("scene_exposure_main_effect", "oracle_prior_main_effect", "interaction")
HISTORICAL_SINGLE_SPLIT = {
    "source": "runs_mpb/paired_contrasts_all.json factorial point estimates (v1 7-class audit)",
    "scene_exposure_main_effect": 0.0227,
    "oracle_prior_main_effect": 0.4183,
    "interaction": -0.0847,
}


def load_categories():
    category, md_inst = {}, {}
    with INVENTORY.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            category[row["stem"]] = row["negative_category_min4"]
            md_inst[row["stem"]] = int(row["md_instances_min4"])
    for stem in category:
        if (category[stem] == POSITIVE) != (md_inst[stem] > 0):
            raise RuntimeError(f"Inventory category/instance mismatch for {stem}")
    return category, md_inst


def load_runtime():
    global np, cvp, rex, at, RAW_DIR, _compute_stats, _stem_hash, INVENTORY, CATEGORY, MD_INST
    if "rex" in globals():
        return
    import numpy as np
    import cv_pipeline as cvp
    import run_experiments as rex
    import audit_toolkit as at
    from prepare_protocol import RAW_DIR, _compute_stats, _stem_hash
    INVENTORY = cvp.AUDIT_DIR / "patch_inventory.csv"
    CATEGORY, MD_INST = load_categories()


def sha_list(stems):
    return hashlib.sha256("\n".join(stems).encode("utf-8")).hexdigest()


def sha_file(path: Path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def require_empty(path: Path):
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Fold design
# --------------------------------------------------------------------------
def choose_removal(k, base, siblings):
    rng = np.random.default_rng(REMOVAL_SEED + k)
    need = Counter(CATEGORY[st] for st in siblings)
    pools = defaultdict(list)
    for st in base:
        pools[CATEGORY[st]].append(st)
    removal = []
    for cat in sorted(need):
        pool = sorted(pools[cat])
        if len(pool) < need[cat]:
            raise RuntimeError(f"fold {k}: base has {len(pool)} {cat} patches, need {need[cat]}")
        if cat == POSITIVE:
            target = sum(MD_INST[st] for st in siblings)
            best, best_gap = None, None
            for _ in range(N_DRAWS):
                draw = [str(st) for st in rng.choice(pool, size=need[cat], replace=False)]
                gap = abs(sum(MD_INST[st] for st in draw) - target)
                if best_gap is None or gap < best_gap:
                    best, best_gap = draw, gap
            removal.extend(best)
        else:
            removal.extend(str(st) for st in rng.choice(pool, size=need[cat], replace=False))
    return sorted(removal)


def fold_design(k):
    train_sc, val_sc, test_sc = cvp.fold_scene_sets(k)
    base = sorted(st for sc in train_sc for st in cvp.SCENE_STEMS[sc])
    val = sorted(st for sc in val_sc for st in cvp.SCENE_STEMS[sc])
    siblings, panel, exposed, single = [], [], [], []
    for sc in sorted(test_sc):
        stems = cvp.SCENE_STEMS[sc]
        if len(stems) < 2:
            single.append(sc)
            panel.extend(stems)
            continue
        exposed.append(sc)
        pos = sorted((st for st in stems if CATEGORY[st] == POSITIVE), key=_stem_hash)
        neg = sorted((st for st in stems if CATEGORY[st] != POSITIVE), key=_stem_hash)
        n_sib = len(stems) // 2
        n_pos = len(pos) // 2
        n_neg = n_sib - n_pos
        if n_neg < 0 or n_neg > len(neg):
            raise RuntimeError(f"fold {k} scene {sc}: invalid sibling stratification")
        siblings.extend(pos[:n_pos] + neg[:n_neg])
        panel.extend(pos[n_pos:] + neg[n_neg:])
    siblings, panel = sorted(siblings), sorted(panel)
    removal = choose_removal(k, base, siblings)
    common = sorted(set(base) - set(removal))
    shared = sorted(set(common) | set(siblings))
    design = {
        "fold": k,
        "test_scenes": sorted(test_sc),
        "exposed_scenes": exposed,
        "single_patch_scenes": single,
        "val": val,
        "panel": panel,
        "siblings": siblings,
        "removal": removal,
        "common": common,
        "train": {"clean": base, "shared": shared},
    }
    validate_design(design, train_sc, val_sc, test_sc)
    return design


def validate_design(d, train_sc, val_sc, test_sc):
    k = d["fold"]
    scene = cvp.STEM_SCENE
    base, shared = d["train"]["clean"], d["train"]["shared"]
    test_all = {st for sc in test_sc for st in cvp.SCENE_STEMS[sc]}
    panel, sib, rem = set(d["panel"]), set(d["siblings"]), set(d["removal"])
    checks = {
        "equal_train_size": len(base) == len(shared),
        "panel_sibling_partition": (panel | sib) == test_all and not (panel & sib),
        "removal_inside_base": rem <= set(base) and len(rem) == len(sib),
        "category_match": Counter(CATEGORY[s] for s in rem) == Counter(CATEGORY[s] for s in sib),
        "clean_scene_disjoint": not ({scene[s] for s in base} & set(test_sc)),
        "shared_exposes_only_exposed": ({scene[s] for s in shared} & set(test_sc)) == set(d["exposed_scenes"]),
        "val_disjoint": not ({scene[s] for s in d["val"]} & (set(train_sc) | set(test_sc))),
        "every_test_scene_in_panel": {scene[s] for s in panel} == set(test_sc),
        "common_is_intersection": set(d["common"]) == set(base) & set(shared),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"fold {k} design checks failed: {failed}")


def all_designs():
    designs = [fold_design(k) for k in range(9)]
    if DESIGN_JSON.exists():
        saved = json.loads(DESIGN_JSON.read_text(encoding="utf-8"))
        if saved != designs:
            raise RuntimeError(f"Recomputed design differs from {DESIGN_JSON}")
    return designs


def design_rows(designs):
    rows = []
    for d in designs:
        exposed_panel = [s for s in d["panel"] if cvp.STEM_SCENE[s] in set(d["exposed_scenes"])]
        rows.append({
            "fold": d["fold"],
            "train_patches": len(d["train"]["clean"]),
            "val_patches": len(d["val"]),
            "test_scenes": len(d["test_scenes"]),
            "exposed_scenes": len(d["exposed_scenes"]),
            "single_patch_scenes": len(d["single_patch_scenes"]),
            "sibling_patches": len(d["siblings"]),
            "sibling_positive_patches": sum(CATEGORY[s] == POSITIVE for s in d["siblings"]),
            "sibling_md_instances": sum(MD_INST[s] for s in d["siblings"]),
            "removal_md_instances": sum(MD_INST[s] for s in d["removal"]),
            "panel_patches": len(d["panel"]),
            "panel_md_instances": sum(MD_INST[s] for s in d["panel"]),
            "exposed_panel_patches": len(exposed_panel),
            "exposed_panel_md_instances": sum(MD_INST[s] for s in exposed_panel),
        })
    return rows


# --------------------------------------------------------------------------
# Oracle arrays and per-cell datasets
# --------------------------------------------------------------------------
def label_path_of(img_path):
    parts = list(Path(img_path).parts)
    for variant in ("images", "images_7ch"):
        if variant in parts:
            parts[parts.index(variant)] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise RuntimeError(f"Cannot derive label path for {img_path}")


def oracle_paths(stem):
    return ORACLE_ROOT / "images_7ch" / f"{stem}.npy", ORACLE_ROOT / "labels" / f"{stem}.txt"


def oracle_prior(stem):
    import cv2
    import rasterio
    mask_path = RAW_DIR / cvp.STEM_SCENE[stem] / f"{stem}_cl.tif"
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    m = dist.max()
    if m > 1e-6:
        return (dist / m).astype(np.float32)
    return dist.astype(np.float32)


def build_oracle_arrays(stems):
    (ORACLE_ROOT / "images_7ch").mkdir(parents=True, exist_ok=True)
    (ORACLE_ROOT / "labels").mkdir(parents=True, exist_ok=True)
    built = 0
    for st in sorted(stems):
        img, lab = oracle_paths(st)
        ndwi_path = cvp.SRC_NPY["c7"][st]
        src_label = label_path_of(ndwi_path)
        if not src_label.is_file():
            raise FileNotFoundError(src_label)
        if not img.exists():
            ndwi = np.load(ndwi_path, allow_pickle=False)
            six = np.load(cvp.SRC_NPY["c6"][st], allow_pickle=False)
            if ndwi.shape[:2] != six.shape[:2] or not np.array_equal(ndwi[..., :6], six, equal_nan=True):
                raise RuntimeError(f"NDWI/6ch mismatch for {st}")
            prior = oracle_prior(st)
            if prior.shape != six.shape[:2] or prior.min() < 0.0 or prior.max() > 1.0:
                raise RuntimeError(f"Invalid oracle prior for {st}")
            arr = np.concatenate([six, prior[..., None]], axis=2).astype(np.float32)
            tmp = img.with_name(f"{st}.tmp.npy")
            with tmp.open("wb") as fh:
                np.save(fh, arr)
            tmp.replace(img)
            built += 1
        if not lab.exists():
            shutil.copyfile(src_label, lab)
        elif lab.read_bytes() != src_label.read_bytes():
            raise RuntimeError(f"Oracle label differs from NDWI label for {st}")
    return built


def fold_stats(d):
    with file_lock(FACT_ROOT / "_locks" / f"fold{d['fold']}_stats.lock", blocking=True):
        return _fold_stats(d)


def _fold_stats(d):
    path = FACT_ROOT / f"fold{d['fold']}_common_stats_7ch.json"
    key = sha_list(d["common"])
    if path.exists():
        stats = json.loads(path.read_text(encoding="utf-8"))
        if stats.get("common_sha256") == key and stats.get("fit_split") == "factorial_train_intersection_only":
            return stats
    stats = _compute_stats([cvp.SRC_NPY["c7"][st] for st in d["common"]], 7, fixed_prior_range=True)
    stats["fit_split"] = "factorial_train_intersection_only"
    stats["fit_scope"] = "scene_cv_factorial_fold"
    stats["fit_contract"] = ("Fitted on base minus removal set (patches present in both clean and shared "
                             "training sets) from NDWI arrays; prior channel fixed to [0, 1]; shared by all four cells.")
    stats["common_sha256"] = key
    write_json(path, stats)
    return stats


def cell_image(cell, stem):
    return cvp.SRC_NPY["c7"][stem] if cell.endswith("_ndwi") else oracle_paths(stem)[0]


def prepare_cell(d, cell, smoke=False):
    tag = f"fold{d['fold']}_{cell}" + ("_smoke" if smoke else "")
    with file_lock(FACT_ROOT / "_locks" / f"{tag}.lock", blocking=True):
        return _prepare_cell(d, cell, smoke)


def _prepare_cell(d, cell, smoke=False):
    exposure = cell.split("_")[0]
    k = d["fold"]
    tag = f"fold{k}_{cell}" + ("_smoke" if smoke else "")
    ddir = FACT_ROOT / tag
    ddir.mkdir(parents=True, exist_ok=True)
    train, val = d["train"][exposure], d["val"]
    if smoke:
        train, val = cvp.cap_stems(train, 2), cvp.cap_stems(val, 2)
    splits = {"train": train, "val": val, "test": d["panel"]}
    counts = {}
    for split, stems in splits.items():
        paths = [Path(cell_image(cell, st)) for st in stems]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{tag}/{split} missing arrays: {missing[:3]}")
        write_consistent(ddir / f"{split}.txt", "\n".join(str(p) for p in paths))
        counts[split] = len(paths)
    write_consistent(ddir / "channel_stats_7ch.json", json.dumps(fold_stats(d), indent=2, ensure_ascii=False))
    metadata = {
        "name": tag,
        "split_kind": "scene_cv_factorial",
        "source_kind": "7ch_ndwi" if cell.endswith("_ndwi") else "7ch_oracle",
        "channels": 7,
        "train_min_pixels_marine_debris": 4,
        "val_test_min_pixels_marine_debris": 4,
        "min_pixels_other_classes": 4,
        "fold": k,
        "val_fold": (k + 1) % 9,
        "exposure": exposure,
        "train_sha256": sha_list(sorted(train)),
        "panel_sha256": sha_list(d["panel"]),
    }
    write_consistent(ddir / "variant_metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
    yaml_text = (
        f"path: {ddir}\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
        f"channels: 7\nnc: 3\nnames:\n"
        + "\n".join(f"  {i}: {n}" for i, n in cvp.NAMES.items())
        + "\n"
    )
    write_consistent(ddir / "cv.yaml", yaml_text)
    return ddir, counts


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def run_cell(d, cell, seed, device, smoke=False):
    tag = f"fold{d['fold']}_{cell}" + ("_smoke" if smoke else "")
    run_name = f"{tag}_s{seed}"
    project = RUNS / "smoke" if smoke else RUNS
    oof_path = project / "oof" / f"{run_name}_oof.json"
    status = run_locked(project / "_locks" / f"{run_name}.lock", project / run_name, oof_path,
                        lambda: _run_cell(d, cell, seed, device, smoke), skip_completed=not smoke)
    print(f"[{status}] {run_name}", flush=True)
    return status


def _run_cell(d, cell, seed, device, smoke=False):
    k = d["fold"]
    ddir, counts = prepare_cell(d, cell, smoke)
    tag = f"fold{k}_{cell}" + ("_smoke" if smoke else "")
    run_name = f"{tag}_s{seed}"
    project = RUNS / "smoke" if smoke else RUNS
    run_dir = project / run_name
    oof_dir = RUNS / "smoke" / "oof" if smoke else OOF_DIR
    oof_path = oof_dir / f"{run_name}_oof.json"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    if oof_path.exists():
        oof_path.unlink()

    hyper = {**rex.LOCKED_HYPERPARAMS}
    if smoke:
        hyper.update(epochs=1, patience=1, workers=0, plots=False)
    with file_lock(RUNS / "_locks" / "initialization.lock", blocking=True):
        init = rex._ensure_initialization()[7]
    overrides = {
        **hyper,
        "model": str(init),
        "data": str(ddir / "cv.yaml"),
        "project": str(project),
        "name": run_name,
        "seed": seed,
        "device": device,
        "exist_ok": True,
    }
    print(f"[train] {run_name} counts={counts}", flush=True)
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
    oof_dir.mkdir(parents=True, exist_ok=True)
    tmp = oof_dir / f"{run_name}_oof.tmp.json"
    rex._write_prediction_ledger(tmp, test_entries)
    tmp.replace(oof_path)
    print(f"[oof] {oof_path.name} entries={len(test_entries)}", flush=True)
    return oof_path


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def load_oof(designs, oof_dir, folds, seeds, cells=CELLS, suffix=""):
    entries = {}
    missing = []
    for k in folds:
        for cell in cells:
            for s in seeds:
                p = oof_dir / f"fold{k}_{cell}{suffix}_s{s}_oof.json"
                if not p.is_file():
                    missing.append(p.name)
    if missing:
        raise RuntimeError(f"Incomplete OOF: missing {len(missing)} files, e.g. {missing[:5]}")
    for k in folds:
        d = designs[k]
        ref = None
        for cell in cells:
            for s in seeds:
                p = oof_dir / f"fold{k}_{cell}{suffix}_s{s}_oof.json"
                e = rex._read_prediction_ledger(p)
                by = {x["stem"]: x for x in e}
                if len(by) != len(e) or set(by) != set(d["panel"]):
                    raise RuntimeError(f"{p.name}: duplicate, missing or extra panel patches")
                for st, x in by.items():
                    if x["scene"] != cvp.STEM_SCENE[st]:
                        raise RuntimeError(f"{p.name}: wrong scene for {st}")
                    if len(x["ground_truth"]) != MD_INST[st]:
                        raise RuntimeError(f"{p.name}: GT count mismatch for {st}")
                if ref is None:
                    ref = by
                else:
                    for st, x in by.items():
                        if not np.array_equal(np.asarray(x["ground_truth"], float),
                                              np.asarray(ref[st]["ground_truth"], float)):
                            raise RuntimeError(f"{p.name}: GT differs across cells/seeds for {st}")
                entries[(k, cell, s)] = e
    return entries


def cell_aps(led, scenes, cells=CELLS):
    return {cell: [at.pooled_ap(lg, scenes) for lg in led[cell]] for cell in cells}


def contrast_values(aps):
    out = {}
    for name, coef in CONTRASTS.items():
        if not all(c in aps for c in coef):
            continue
        n = len(next(iter(aps.values())))
        per_seed = [sum(w * aps[c][i] for c, w in coef.items()) for i in range(n)]
        out[name] = float(np.mean(per_seed))
    return out


def bootstrap(led, scenes, n_boot, seed):
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    skipped = 0
    for _ in range(n_boot):
        sample = rng.choice(scenes, size=len(scenes), replace=True)
        aps = cell_aps(led, sample)
        if any(np.isnan(v) for vals in aps.values() for v in vals):
            skipped += 1
            continue
        for cell, vals in aps.items():
            draws[f"AP_{cell}"].append(float(np.mean(vals)))
        for name, v in contrast_values(aps).items():
            draws[name].append(v)
    ci = {name: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for name, v in draws.items()}
    return ci, skipped


def summarize_panel(led, scenes, n_boot, seed):
    aps = cell_aps(led, scenes)
    point = contrast_values(aps)
    ci, skipped = bootstrap(led, scenes, n_boot, seed)
    out = {
        "n_scenes": len(scenes),
        "n_gt": int(sum(led["clean_ndwi"][0][sc][2] for sc in scenes)),
        "cells": {cell: {"per_seed_AP": [round(v, 4) for v in vals],
                         "mean_AP": round(float(np.mean(vals)), 4),
                         "ci95": [round(x, 4) for x in ci[f"AP_{cell}"]]}
                  for cell, vals in aps.items()},
        "contrasts": {},
        "bootstrap_replicates": n_boot,
        "bootstrap_skipped_nan": skipped,
        "bootstrap_seed": seed,
    }
    for name, v in point.items():
        lo, hi = ci[name]
        out["contrasts"][name] = {
            "point": round(v, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            f"ci_vs_margin_{MARGIN}": at.ci_vs_margin(lo, hi, MARGIN),
        }
    return out


def jackknife(led, scenes):
    out = {}
    for name in PRIMARY_CONTRASTS:
        vals = {}
        for sc in scenes:
            rest = [s for s in scenes if s != sc]
            vals[sc] = contrast_values(cell_aps(led, rest))[name]
        lo = min(vals, key=vals.get)
        hi = max(vals, key=vals.get)
        out[name] = {"min": {"scene": lo, "value": round(vals[lo], 4)},
                     "max": {"scene": hi, "value": round(vals[hi], 4)}}
    return out


def analyze(n_boot):
    require_empty(ANALYSIS_OUT)
    designs = all_designs()
    entries = load_oof(designs, OOF_DIR, range(9), SEEDS)
    led = {cell: [cvp.entries_to_scene_ledger([x for k in range(9) for x in entries[(k, cell, s)]])
                  for s in SEEDS] for cell in CELLS}
    all_scenes = sorted(led["clean_ndwi"][0])
    exposed = sorted(sc for d in designs for sc in d["exposed_scenes"])
    if len(all_scenes) != 63 or not set(exposed) <= set(all_scenes):
        raise RuntimeError("Panel ledger must cover all 63 scenes")
    for cell in CELLS:
        for lg in led[cell]:
            if sorted(lg) != all_scenes:
                raise RuntimeError(f"{cell}: scene set differs")

    per_fold_rows = []
    fold_means = {cell: [] for cell in CELLS}
    for d in designs:
        scenes = d["exposed_scenes"]
        n_gt = sum(led["clean_ndwi"][0][sc][2] for sc in scenes)
        for cell in CELLS:
            fold_led = [cvp.entries_to_scene_ledger(entries[(d["fold"], cell, s)]) for s in SEEDS]
            aps = [at.pooled_ap(lg, scenes) if n_gt else float("nan") for lg in fold_led]
            for s, ap in zip(SEEDS, aps):
                per_fold_rows.append({"fold": d["fold"], "cell": cell, "seed": s,
                                      "exposed_scenes": len(scenes), "n_gt": n_gt, "AP": ap})
            if n_gt:
                fold_means[cell].append(float(np.mean(aps)))
    fold_mean_ap = {cell: round(float(np.mean(v)), 4) for cell, v in fold_means.items()}
    fold_mean_contrasts = {name: round(sum(w * fold_mean_ap[c] for c, w in coef.items()), 4)
                           for name, coef in CONTRASTS.items()}

    report = {
        "plan_path": str(PLAN_PATH),
        "plan_sha256": sha_file(PLAN_PATH) if PLAN_PATH.exists() else None,
        "script_sha256": sha_file(Path(__file__)),
        "design_sha256": sha_file(DESIGN_JSON),
        "integrity": {"status": "passed", "n_runs": len(entries), "folds": 9,
                      "cells": list(CELLS), "seeds": list(SEEDS),
                      "panel_patches": sum(len(d["panel"]) for d in designs),
                      "same_GT_across_cells_and_seeds": True},
        "statistic": "Mean across paired seeds of globally pooled OOF debris AP50 (101-point, IoU>=0.5)",
        "primary_panel_exposed_scenes": summarize_panel(led, exposed, n_boot, BOOT_SEED),
        "secondary_panel_all_scenes": summarize_panel(led, all_scenes, n_boot, BOOT_SEED),
        "secondary_fold_mean_AP_exposed": {"cells": fold_mean_ap, "contrasts": fold_mean_contrasts},
        "secondary_jackknife_exposed": jackknife(led, exposed),
        "design_summary": design_rows(designs),
        "historical_single_split_reference": HISTORICAL_SINGLE_SPLIT,
        "conditioning": ("Scene resampling conditions on the fitted models and this fold allocation; "
                         "pooled AP ranks scores across separately fitted fold models."),
    }
    write_json(ANALYSIS_OUT / "summary.json", report)
    with (ANALYSIS_OUT / "per_fold_per_seed.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_fold_rows[0]))
        w.writeheader()
        w.writerows(per_fold_rows)
    print(json.dumps({k: report[k] for k in ("integrity", "primary_panel_exposed_scenes")},
                     indent=2, ensure_ascii=False))
    print("written:", ANALYSIS_OUT)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------
def run_design():
    require_empty(DESIGN_OUT)
    designs = all_designs()
    write_json(DESIGN_JSON, designs)
    needed = sorted({st for d in designs for st in d["panel"] + d["siblings"] + d["val"] + d["train"]["clean"]})
    built = build_oracle_arrays(needed)
    for d in designs:
        fold_stats(d)
    rows = design_rows(designs)
    with (DESIGN_OUT / "design_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(DESIGN_OUT / "design_check.json", {
        "design_json": str(DESIGN_JSON),
        "design_sha256": sha_file(DESIGN_JSON),
        "oracle_arrays_total": len(needed),
        "oracle_arrays_built_now": built,
        "totals": {key: sum(r[key] for r in rows) for key in rows[0] if key != "fold"},
    })
    print(json.dumps(rows, indent=2))
    print("written:", DESIGN_OUT)


def run_smoke(device):
    designs = all_designs()
    d = designs[0]
    for cell in CELLS:
        if run_cell(d, cell, 0, device, smoke=True) == "busy":
            raise RuntimeError(f"Incomplete smoke: {cell} is busy")
    entries = load_oof(designs, RUNS / "smoke" / "oof", [0], (0,), suffix="_smoke")
    for cell in CELLS:
        led = cvp.entries_to_scene_ledger(entries[(0, cell, 0)])
        print(f"SMOKE {cell}: panel AP(fold0 exposed) = "
              f"{at.pooled_ap(led, d['exposed_scenes']):.4f} entries={len(entries[(0, cell, 0)])}")
    print("SMOKE OK")


def initialize_worker():
    load_runtime()
    all_designs()


def run_task(task):
    d, cell, seed, device = task
    return run_cell(d, cell, seed, device)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    ap.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    return ap.parse_args(argv)


def main():
    args = parse_args()
    if args.analyze:
        if not DESIGN_JSON.exists():
            raise SystemExit("Run --design first")
        load_runtime()
        analyze(args.n_boot)
        return
    with file_lock(RUNS / "_locks" / "scheduler.lock"):
        reject_other_schedulers()
        run_selected(args)


def run_selected(args):
    if args.design:
        load_runtime()
        run_design()
        return
    if not DESIGN_JSON.exists():
        raise SystemExit("Run --design first")
    if args.smoke:
        load_runtime()
        run_smoke(args.device)
        return
    designs = json.loads(DESIGN_JSON.read_text(encoding="utf-8"))
    tasks = [(*task, args.device) for task in build_tasks(designs, CELLS, args.seed)]
    schedule(tasks, run_task, args.jobs, initializer=initialize_worker)
    if set(args.seed) == set(SEEDS):
        print("ALL 108 RUNS COMPLETE. Next: --analyze")
    else:
        print(f"SELECTED {len(tasks)} RUNS COMPLETE. Analysis still requires all 108 runs.")


if __name__ == "__main__":
    main()
