"""
stat_revision_analysis.py
Statistical-inference revision for the MPB marine-debris audit.

Runs on the EXISTING full-scene test outputs (10 scenes, 3 seeds):
  - build scene ledgers scene -> (scores, tp, n_gt) from test_predictions_locked.json
  - scene-clustered bootstrap, paired scene-swap permutation, leave-one-scene jackknife
  - equivalence reading against a predeclared smallest effect of interest (margin)
  - small-sample Type-I-error simulation (10 scenes, true diff 0, 150 replicates)
  - balanced 9-fold leave-scene-out fold definitions over all 63 scenes
Outputs to runs_mpb/stat_revision/.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\OceanEdge_Group\Marine-Debris-Detection-main")
sys.path.insert(0, str(ROOT / "scripts" / "mpb"))
import audit_toolkit as at  # noqa: E402

FULL = ROOT / "runs_mpb" / "full"
AUDIT = ROOT / "data" / "processed" / "marida_mpb_v1" / "audit"
OUT = ROOT / "runs_mpb" / "stat_revision"
OUT.mkdir(parents=True, exist_ok=True)

CELL_PREFIX = {
    "C6": "mpb__full_scene_6ch",
    "C7": "mpb__full_scene_ndwi7",
    "C7sh": "mpb__full_scene_shuffled7",
}
MARGIN = 0.05
MATCH_IOU = 0.5
SEEDS = (0, 1, 2)


# --------------------------------------------------------------------------
# Ledger construction
# --------------------------------------------------------------------------
def box_iou_one_vs_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def ledger_from_predictions(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    ledger: dict[str, tuple] = {}
    for e in data["entries"]:
        scene = e["scene"]
        preds = np.asarray(e.get("predictions_xyxy_conf_cls", []), float).reshape(-1, 6)
        preds = preds[preds[:, 5] == 0.0]  # cls 0 = Marine_Debris
        gts = np.asarray(e.get("ground_truth_xyxy", []), float).reshape(-1, 4)

        tp = np.zeros(len(preds))
        used = np.zeros(len(gts), bool)
        for i in np.argsort(-preds[:, 4], kind="stable"):
            if not len(gts):
                break
            avail = np.where(~used)[0]
            if not len(avail):
                break
            ious = box_iou_one_vs_many(preds[i, :4], gts[avail])
            j = int(np.argmax(ious))
            if ious[j] >= MATCH_IOU:
                tp[i] = 1.0
                used[avail[j]] = True

        s0, t0, n0 = ledger.get(scene, (np.array([]), np.array([]), 0))
        ledger[scene] = (
            np.concatenate([s0, preds[:, 4]]) if len(s0) else preds[:, 4].copy(),
            np.concatenate([t0, tp]) if len(t0) else tp.copy(),
            n0 + len(gts),
        )
    return ledger


def load_cell(cell: str) -> list[dict]:
    prefix = CELL_PREFIX[cell]
    out = []
    for seed in SEEDS:
        p = FULL / f"{prefix}__s{seed}" / "test_predictions_locked.json"
        out.append(ledger_from_predictions(p))
    return out


def mean_ap_over_seeds(ledgers: list[dict], scenes) -> float:
    return float(np.mean([at.pooled_ap(lg, scenes) for lg in ledgers]))


# --------------------------------------------------------------------------
# Existing 10-scene analysis
# --------------------------------------------------------------------------
def analyze_existing():
    C6 = load_cell("C6")
    C7 = load_cell("C7")
    C7sh = load_cell("C7sh")
    scenes = sorted(C6[0])

    # Baseline check: reproduce reported ledger AP per seed
    ap6 = [at.pooled_ap(lg, scenes) for lg in C6]
    ap7 = [at.pooled_ap(lg, scenes) for lg in C7]
    apsh = [at.pooled_ap(lg, scenes) for lg in C7sh]

    # Statistic defined as C7 - C6 (the predeclared channel contrast)
    pt, (lo, hi), draws = at.scene_bootstrap(C7, C6, B=10000, seed=20261012)
    obs, p_perm, _ = at.paired_scene_swap(C7, C6)
    full_jk, jk = at.scene_jackknife(C7, C6)

    jk_sorted = sorted(jk.items(), key=lambda kv: kv[1])
    margin_class = at.ci_vs_margin(lo, hi, MARGIN)

    result = {
        "n_scenes": len(scenes),
        "scenes": scenes,
        "per_seed_AP": {
            "C6": [round(x, 4) for x in ap6],
            "C7": [round(x, 4) for x in ap7],
            "C7sh": [round(x, 4) for x in apsh],
        },
        "mean_AP": {
            "C6": round(float(np.mean(ap6)), 4),
            "C7": round(float(np.mean(ap7)), 4),
            "C7sh": round(float(np.mean(apsh)), 4),
        },
        "contrast_C7_minus_C6": {
            "point": round(pt, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "permutation_p_two_sided": round(p_perm, 4),
            "jackknife_full": round(full_jk, 4),
            "jackknife_min": {"scene": jk_sorted[0][0], "diff": round(jk_sorted[0][1], 4)},
            "jackknife_max": {"scene": jk_sorted[-1][0], "diff": round(jk_sorted[-1][1], 4),
                              "n_gt_in_scene": int(C6[0][jk_sorted[-1][0]][2])},
            "ci_vs_margin_%s" % MARGIN: margin_class,
        },
        "jackknife_all": {sc: round(v, 4) for sc, v in sorted(jk.items())},
    }
    return result, (C6, C7, C7sh)


# --------------------------------------------------------------------------
# Type-I-error simulation
# --------------------------------------------------------------------------
def type_i_error_simulation(reps: int = 150, n_scenes: int = 10, boot_B: int = 1000):
    rng = np.random.default_rng(20261012)
    scenes = [f"S{i}" for i in range(n_scenes)]
    quality = 0.6
    boot_reject = 0
    perm_reject = 0
    boot_rows = []
    for r in range(reps):
        A = [at._synthetic_ledger(rng, scenes, quality)]
        B = [at._synthetic_ledger(rng, scenes, quality)]
        _, (lo, hi), _ = at.scene_bootstrap(A, B, B=boot_B, seed=int(rng.integers(1 << 30)))
        _, p, _ = at.paired_scene_swap(A, B)
        b_rej = bool(lo > 0 or hi < 0)
        q_rej = bool(p < 0.05)
        boot_reject += b_rej
        perm_reject += q_rej
        boot_rows.append({"rep": r, "bootstrap_excludes_0": b_rej, "perm_p_lt_0.05": q_rej,
                          "ci": [round(lo, 4), round(hi, 4)], "perm_p": round(p, 4)})
    n = reps
    se = float(np.sqrt(0.05 * 0.95 / n))
    return {
        "replicates": reps,
        "n_scenes": n_scenes,
        "true_difference": 0.0,
        "nominal_alpha": 0.05,
        "percentile_bootstrap_rejection_rate": round(boot_reject / n, 4),
        "paired_permutation_rejection_rate": round(perm_reject / n, 4),
        "binomial_se_at_p0.05": round(se, 4),
        "note": "Small n; rates fluctuate by roughly +/-2 percentage points. Supports adding "
                "the permutation test; it is not proof that bootstrap is invalid.",
    }, boot_rows


# --------------------------------------------------------------------------
# 9-fold definitions over all 63 scenes
# --------------------------------------------------------------------------
def build_scene_counts():
    scene_min4 = defaultdict(int)
    scene_patches = defaultdict(set)

    all_scenes = set()
    with (AUDIT / "patch_inventory.csv").open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            sc = row["scene"]
            all_scenes.add(sc)
            scene_patches[sc].add(row["stem"])
            scene_min4[sc] += int(row["md_instances_min4"])

    comp_min4 = defaultdict(int)
    with (AUDIT / "marine_debris_components.csv").open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["class_name"] == "Marine_Debris" and int(row["area_pixels"]) >= 4:
                comp_min4[row["scene"]] += 1

    mismatch = {sc: (scene_min4[sc], comp_min4.get(sc, 0))
                for sc in all_scenes if scene_min4[sc] != comp_min4.get(sc, 0)}
    counts = {sc: scene_min4[sc] for sc in all_scenes}
    return counts, scene_patches, mismatch


def define_folds():
    counts, scene_patches, mismatch = build_scene_counts()
    folds = at.grouped_scene_folds(counts, k=9, seed=20261012)
    fold_def = []
    for i, scs in enumerate(folds):
        fold_def.append({
            "fold": i,
            "test_scenes": sorted(scs),
            "val_fold": (i + 1) % 9,
            "n_scenes": len(scs),
            "n_patches": int(sum(len(scene_patches[s]) for s in scs)),
            "n_debris_min4": int(sum(counts[s] for s in scs)),
        })
    return {
        "n_scenes_total": len(counts),
        "n_folds": 9,
        "total_debris_min4": int(sum(counts.values())),
        "inventory_vs_components_mismatch": mismatch,
        "fold_assignment_rule": "fold i = test; fold (i+1)%9 = validation; remaining 7 folds = train",
        "folds": fold_def,
    }


def main():
    existing, _ = analyze_existing()
    typei, typei_rows = type_i_error_simulation()
    folds = define_folds()

    payload = {"existing_10_scene": existing, "type_i_simulation": typei, "cv_folds": folds}
    (OUT / "stat_revision_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "type_i_simulation_per_replicate.json").write_text(
        json.dumps(typei_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # fold manifest as csv
    with (OUT / "cv9_fold_assignment.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "role", "scene", "n_patches_in_scene", "n_debris_min4_in_scene"])
        counts, scene_patches, _ = build_scene_counts()
        for fd in folds["folds"]:
            for s in fd["test_scenes"]:
                w.writerow([fd["fold"], "test", s, len(scene_patches[s]), counts[s]])

    print("== per-seed AP (ledger, recomputed) ==")
    print("C6 :", existing["per_seed_AP"]["C6"], " mean", existing["mean_AP"]["C6"])
    print("C7 :", existing["per_seed_AP"]["C7"], " mean", existing["mean_AP"]["C7"])
    print("C7sh:", existing["per_seed_AP"]["C7sh"], " mean", existing["mean_AP"]["C7sh"])
    c = existing["contrast_C7_minus_C6"]
    print("\n== contrast C7-C6 ==")
    print("point", c["point"], "CI", c["ci95"], "perm p", c["permutation_p_two_sided"])
    print("jackknife full", c["jackknife_full"], "| drop-min", c["jackknife_min"],
          "| drop-max", c["jackknife_max"])
    print("ci_vs_margin:", c["ci_vs_margin_%s" % MARGIN])
    print("\n== Type I simulation ==")
    print("bootstrap reject", typei["percentile_bootstrap_rejection_rate"],
          " permutation reject", typei["paired_permutation_rejection_rate"],
          " binomial SE", typei["binomial_se_at_p0.05"])
    print("\n== folds ==")
    print("scenes", folds["n_scenes_total"], "total debris MIN4", folds["total_debris_min4"],
          "mismatch", folds["inventory_vs_components_mismatch"])
    for fd in folds["folds"]:
        print("fold %d: scenes=%d patches=%d debris=%d" %
              (fd["fold"], fd["n_scenes"], fd["n_patches"], fd["n_debris_min4"]))


if __name__ == "__main__":
    main()
