"""False-positive sources and missed-detection strata for the 9-fold scene CV OOF.

Read-only with respect to existing results; writes only into --out.

Systems: YOLO C6 / C7 x 3 seeds (validated OOF, debris class 0 only) and the
pre-specified FDI baseline OOF (runs_v2/_tmp_eval/fdi_baseline/fdi_oof.json).

Matching: same greedy per-patch matcher as cv_pipeline.match_patch_tp
(IoU>=0.5 primary, IoU>=0.1 secondary); duplicates count as FP.

Operating points:
  * top_ngt (primary): inside every CV fold's test scenes, the n_gt(fold)
    highest-scoring predictions (R-precision point; one cut per fold, so no
    cross-fold score calibration is assumed);
  * all: every OOF prediction (conf >= 0.001 for YOLO; every FDI component).

FP attribution:
  (a) patch level: negative_category_min4 of the patch (patch_inventory.csv);
  (b) pixel level, inside the FP box mapped to the 256 grid (*_cl.tif):
      near_gt_iou_ge_0.1 if max IoU with any MIN4 GT box of the patch >= 0.1,
      else other_debris_pixels if the box holds any class-1 pixel,
      else the dominant annotated group among classes 2-15,
      else unlabeled_only.

Missed detections: GT boxes are linked one-to-one to
marine_debris_components.csv (area >= 4) through the label-builder geometry
(pad 2 px, clip, x2 to the 512 grid); recall is stratified by component area
(4-7 / 8-15 / >=16 px) and dominant annotation confidence (1 high, 2 moderate,
3 low), at the top_ngt point and at maximum recall (all predictions).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "mpb"))

import audit_toolkit as at  # noqa: E402
import cv_pipeline as cv  # noqa: E402
import v2_fdi_baseline as fb  # noqa: E402
import v2_relaxed_match as rm  # noqa: E402

SEEDS = tuple(cv.FULL_SEEDS)
CHANNELS = ("c6", "c7")
FOLDS = tuple(range(9))
TOL = 1e-4
GRID_SCALE = 0.5  # 512 input grid -> 256 original grid
PATCH = 256
IOU_THRS = (0.50, 0.10)
NEAR_GT_IOU = 0.10
FP_DEFS = ("top_ngt", "all")
RECALL_DEFS = ("top_ngt", "max_recall")
RAW_GROUP = {1: "debris", 2: "algae_organic", 3: "algae_organic", 4: "algae_organic",
             5: "ship", 6: "cloud_shadow", 13: "cloud_shadow",
             9: "foam_waves_wakes", 12: "foam_waves_wakes", 14: "foam_waves_wakes",
             7: "water", 8: "water", 10: "water", 11: "water", 15: "water"}
ANNOT_GROUPS = ("algae_organic", "ship", "foam_waves_wakes", "cloud_shadow", "water")
PIXEL_CATS = ("near_gt_iou_ge_0.1", "other_debris_pixels") + ANNOT_GROUPS + ("unlabeled_only",)
PATCH_CATS = ("marine_debris_positive", "subthreshold_marine_debris",
              "hard_confounder_algae_or_ship", "visual_confounder_foam_waves_wakes",
              "environmental_water_background", "cloud_or_shadow", "unannotated_or_other")
CONF_NAMES = {1: "high", 2: "moderate", 3: "low"}


def area_stratum(area: int) -> str:
    if area < 8:
        return "area_4-7"
    if area < 16:
        return "area_8-15"
    return "area_ge16"


AREA_STRATA = ("area_4-7", "area_8-15", "area_ge16")


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    aa = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    ab = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-12)


def greedy_match(preds: np.ndarray, gts: np.ndarray, thr: float):
    """Replicates cv_pipeline.match_patch_tp; also returns, per GT, the index of
    the prediction matched to it (-1 if none)."""
    tp = np.zeros(len(preds))
    gt_pred = np.full(len(gts), -1, dtype=np.int64)
    used = np.zeros(len(gts), bool)
    for i in np.argsort(-preds[:, 4], kind="stable"):
        avail = np.where(~used)[0]
        if not len(avail):
            break
        ious = cv.box_iou_one_vs_many(preds[i, :4], gts[avail])
        j = int(np.argmax(ious))
        if ious[j] >= thr:
            tp[i] = 1.0
            used[avail[j]] = True
            gt_pred[avail[j]] = i
    return tp, gt_pred


def padded_box_512(x0: int, y0: int, x1: int, y1: int, pad: int = 2) -> tuple:
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(PATCH - 1, x1 + pad), min(PATCH - 1, y1 + pad)
    return (2.0 * x0, 2.0 * y0, 2.0 * (x1 + 1), 2.0 * (y1 + 1))


def box_pixels(cl: np.ndarray, box: np.ndarray) -> np.ndarray:
    x0 = int(np.floor(max(0.0, box[0] * GRID_SCALE)))
    y0 = int(np.floor(max(0.0, box[1] * GRID_SCALE)))
    x1 = int(np.ceil(min(float(PATCH), box[2] * GRID_SCALE)))
    y1 = int(np.ceil(min(float(PATCH), box[3] * GRID_SCALE)))
    x0, y0 = min(x0, PATCH - 1), min(y0, PATCH - 1)
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    return cl[y0:y1, x0:x1]


GROUP_ORDER = ("unlabeled", "debris") + ANNOT_GROUPS
GROUP_LUT = np.array([GROUP_ORDER.index(RAW_GROUP.get(c, "unlabeled")) for c in range(16)],
                     dtype=np.int64)


def pixel_category(pix: np.ndarray, max_iou: float) -> tuple:
    counts = np.bincount(GROUP_LUT[pix.ravel()], minlength=len(GROUP_ORDER))
    present = {g: bool(counts[GROUP_ORDER.index(g)] > 0) for g in GROUP_ORDER[1:]}
    if max_iou >= NEAR_GT_IOU:
        return "near_gt_iou_ge_0.1", present
    if counts[1] > 0:
        return "other_debris_pixels", present
    group_counts = counts[2:]
    if group_counts.max() > 0:
        return ANNOT_GROUPS[int(np.argmax(group_counts))], present
    return "unlabeled_only", present


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw/MARIDA/patches")
    p.add_argument("--fdi-dir", default="runs_v2/_tmp_eval/fdi_baseline")
    p.add_argument("--ref-json", default="runs_cv/cv_stat_results.json")
    p.add_argument("--out", default="runs_v2/_tmp_eval/error_analysis")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir, fdi_dir, out = Path(args.raw_dir), Path(args.fdi_dir), Path(args.out)
    fb.guard_output_dir(out)

    print("[1/8] loading validated YOLO OOF and FDI OOF ...", flush=True)
    entries_by_run, _ = cv.load_validated_oof(SEEDS, smoke=False)
    scene_fold = {sc: k for k in FOLDS for sc in cv.FOLDS[k]}
    gt_by_stem, scene_by_stem = {}, {}
    for k in FOLDS:
        for e in entries_by_run[(k, "c6", SEEDS[0])]:
            gt_by_stem[e["stem"]] = np.asarray(e["ground_truth"], float).reshape(-1, 4)
            scene_by_stem[e["stem"]] = e["scene"]
    stems = sorted(gt_by_stem)
    if len(stems) != 1381 or sum(len(v) for v in gt_by_stem.values()) != 212:
        raise SystemExit("[error] expected 1381 patches / 212 GT boxes")

    systems = {}
    for ch in CHANNELS:
        for seed in SEEDS:
            preds = {}
            for k in FOLDS:
                for e in entries_by_run[(k, ch, seed)]:
                    p = np.asarray(e["predictions"], float).reshape(-1, 6)
                    preds[e["stem"]] = p[p[:, 5] == 0.0]
            systems[f"{ch}_s{seed}"] = preds
    fdi_oof = json.loads((fdi_dir / "fdi_oof.json").read_text(encoding="utf-8"))
    fdi_preds = {}
    for e in fdi_oof:
        st = e["stem"]
        if st in fdi_preds or st not in gt_by_stem or e["scene"] != scene_by_stem[st]:
            raise SystemExit(f"[error] FDI OOF: duplicate/unknown stem or wrong scene {st}")
        g = np.asarray(e["ground_truth_xyxy"], float).reshape(-1, 4)
        if g.shape != gt_by_stem[st].shape or not np.allclose(g, gt_by_stem[st], atol=1e-3):
            raise SystemExit(f"[error] FDI OOF GT differs from YOLO OOF GT for {st}")
        p = np.asarray(e["predictions_xyxy_conf_cls"], float).reshape(-1, 6)
        fdi_preds[st] = p[p[:, 5] == 0.0]
    if sorted(fdi_preds) != stems:
        raise SystemExit("[error] FDI OOF does not cover every patch exactly once")
    systems["fdi"] = fdi_preds
    system_names = list(systems)

    print("[2/8] reference AP checks (YOLO cv_stat_results, FDI summary) ...", flush=True)
    expected = rm.load_reference_merged_ap(Path(args.ref_json))
    for ch in CHANNELS:
        for si, seed in enumerate(SEEDS):
            led = cv.entries_to_scene_ledger(
                [e for k in FOLDS for e in entries_by_run[(k, ch, seed)]])
            ap = at.pooled_ap(led, sorted(led))
            if not np.isclose(ap, expected[ch][si], atol=TOL, rtol=0.0):
                raise SystemExit(f"[error] {ch} seed {seed}: AP {ap} != {expected[ch][si]}")
    fdi_summary = json.loads((fdi_dir / "summary.json").read_text(encoding="utf-8"))
    fdi_led = cv.entries_to_scene_ledger(
        [{"scene": scene_by_stem[st], "predictions": fdi_preds[st],
          "ground_truth": gt_by_stem[st]} for st in stems])
    fdi_ap = at.pooled_ap(fdi_led, sorted(fdi_led))
    if not np.isclose(fdi_ap, fdi_summary["fdi_results"]["iou_0.50"]["pooled_AP"],
                      atol=TOL, rtol=0.0):
        raise SystemExit(f"[error] FDI AP {fdi_ap} does not reproduce fdi_baseline summary")
    print("      reproduced", flush=True)

    print("[3/8] patch inventory, *_cl.tif and debris components ...", flush=True)
    patch_cat, inv_pixels = {}, {}
    with (cv.AUDIT_DIR / "patch_inventory.csv").open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        pix_cols = {}
        for name in reader.fieldnames:
            if name.startswith("raw_pixels_class_"):
                pix_cols[int(name.split("_")[3])] = name
        if sorted(pix_cols) != list(range(16)):
            raise SystemExit("[error] patch_inventory.csv: raw_pixels_class_0..15 not found")
        for row in reader:
            patch_cat[row["stem"]] = row["negative_category_min4"]
            inv_pixels[row["stem"]] = np.array([int(row[pix_cols[c]]) for c in range(16)])
    if sorted(patch_cat) != stems or not set(patch_cat.values()) <= set(PATCH_CATS):
        raise SystemExit("[error] patch_inventory.csv stems/categories unexpected")
    cl_by_stem = {}
    class_pixels = np.zeros(16, np.int64)
    for i, st in enumerate(stems, start=1):
        path = raw_dir / scene_by_stem[st] / f"{st}_cl.tif"
        with rasterio.open(path) as src:
            arr = src.read(1)
        cl = np.rint(arr).astype(np.int64)
        if cl.shape != (PATCH, PATCH) or cl.min() < 0 or cl.max() > 15:
            raise SystemExit(f"[error] {path}: unexpected shape or class values")
        counts = np.bincount(cl.ravel(), minlength=16)
        if not np.array_equal(counts, inv_pixels[st]):
            raise SystemExit(f"[error] {st}: class pixel counts differ from patch_inventory")
        class_pixels += counts
        cl_by_stem[st] = cl.astype(np.uint8)
        if i % 300 == 0 or i == len(stems):
            print(f"      {i}/{len(stems)} label rasters", flush=True)
    total_pixels = float(class_pixels.sum())
    pixel_ref = {"unlabeled_only": class_pixels[0] / total_pixels,
                 "debris": class_pixels[1] / total_pixels}
    for g in ANNOT_GROUPS:
        pixel_ref[g] = sum(class_pixels[c] for c, gg in RAW_GROUP.items() if gg == g) / total_pixels
    pixel_ref["near_gt_iou_ge_0.1"] = pixel_ref["debris"]
    pixel_ref["other_debris_pixels"] = pixel_ref["debris"]

    comps_by_stem = {}
    with (cv.AUDIT_DIR / "marine_debris_components.csv").open(encoding="utf-8-sig",
                                                               newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["original_class"]) != 1 or int(row["area_pixels"]) < 4:
                continue
            box = padded_box_512(int(row["x_min"]), int(row["y_min"]),
                                 int(row["x_max"]), int(row["y_max"]))
            comps_by_stem.setdefault(row["stem"], []).append(
                {"box": np.array(box), "area": int(row["area_pixels"]),
                 "conf": int(row["dominant_confidence"]), "id": row["component_id"]})
    gt_meta = {}
    n_ambiguous = 0
    for st in stems:
        gts = gt_by_stem[st]
        comps = list(comps_by_stem.get(st, []))
        if len(comps) != len(gts):
            raise SystemExit(f"[error] {st}: {len(comps)} MIN4 components vs {len(gts)} GT")
        meta = []
        for g in gts:
            hits = [ci for ci, c in enumerate(comps) if np.allclose(c["box"], g, atol=1e-3)]
            if not hits:
                raise SystemExit(f"[error] {st}: GT box {g.tolist()} has no component")
            if len(hits) > 1:
                n_ambiguous += 1
            c = comps.pop(hits[0])
            if c["conf"] not in CONF_NAMES:
                raise SystemExit(f"[error] component {c['id']}: confidence {c['conf']}")
            meta.append({"area": c["area"], "conf": c["conf"], "id": c["id"]})
        gt_meta[st] = meta
    n_linked = sum(len(v) for v in gt_meta.values())
    if n_linked != 212:
        raise SystemExit(f"[error] linked {n_linked} GT to components, expected 212")
    print(f"      212 GT linked to components ({n_ambiguous} identical-box ties)", flush=True)

    print("[4/8] matching (IoU 0.5 / 0.1) with matcher-equivalence checks ...", flush=True)
    match10_ref = rm.make_matcher("iou", 0.10, 1.0)
    res = {}
    for name in system_names:
        per = {}
        tp_total = {thr: 0.0 for thr in IOU_THRS}
        for st in stems:
            p, g = systems[name][st], gt_by_stem[st]
            r = {"preds": p}
            for thr in IOU_THRS:
                tp, gt_pred = greedy_match(p, g, thr)
                ref_tp = cv.match_patch_tp(p, g) if thr == 0.50 else match10_ref(p, g)
                if not np.array_equal(tp, ref_tp):
                    raise SystemExit(f"[error] {name} {st} IoU{thr}: matcher differs")
                if int((gt_pred >= 0).sum()) != int(tp.sum()):
                    raise SystemExit(f"[error] {name} {st}: GT/TP count mismatch")
                r[f"tp_{thr}"], r[f"gtp_{thr}"] = tp, gt_pred
                tp_total[thr] += tp.sum()
            r["max_iou"] = (pairwise_iou(p[:, :4], g).max(axis=1) if len(g)
                            else np.zeros(len(p)))
            per[st] = r
        res[name] = per
        print(f"      {name}: TP@0.5={int(tp_total[0.50])} TP@0.1={int(tp_total[0.10])} "
              f"of 212", flush=True)

    print("[5/8] per-fold top-n_gt operating point ...", flush=True)
    stems_by_fold = {k: [st for st in stems if scene_fold[scene_by_stem[st]] == k]
                     for k in FOLDS}
    op_rows = []
    for name in system_names:
        per = res[name]
        for st in stems:
            per[st]["sel"] = np.zeros(len(per[st]["preds"]), bool)
        for k in FOLDS:
            n_gt = int(sum(len(gt_by_stem[st]) for st in stems_by_fold[k]))
            owner = [(st, i) for st in stems_by_fold[k] for i in range(len(per[st]["preds"]))]
            scores = np.array([per[st]["preds"][i, 4] for st, i in owner])
            order = np.argsort(-scores, kind="stable")[:n_gt]
            for idx in order:
                st, i = owner[idx]
                per[st]["sel"][i] = True
            n_tp = int(sum(per[owner[idx][0]]["tp_0.5"][owner[idx][1]] for idx in order))
            op_rows.append({"system": name, "fold": k, "n_gt": n_gt,
                            "n_predictions_fold": len(owner), "n_selected": len(order),
                            "score_cutoff": float(scores[order[-1]]) if len(order) else "",
                            "n_tp_iou50": n_tp, "n_fp_iou50": len(order) - n_tp,
                            "precision_iou50": n_tp / len(order) if len(order) else ""})

    print("[6/8] false-positive attribution ...", flush=True)
    n_patches_cat = {c: sum(1 for st in stems if patch_cat[st] == c) for c in PATCH_CATS}
    fp_patch_rows, fp_pixel_rows = [], []
    fp_patch_share = {}
    fp_pixel_share = {}
    for name in system_names:
        per = res[name]
        for fp_def in FP_DEFS:
            cat_n = {c: 0 for c in PATCH_CATS}
            pix_n = {c: 0 for c in PIXEL_CATS}
            presence = {g: 0 for g in ("debris",) + ANNOT_GROUPS}
            n_fp = 0
            for st in stems:
                r = per[st]
                fp_mask = r["tp_0.5"] == 0.0
                if fp_def == "top_ngt":
                    fp_mask = fp_mask & r["sel"]
                idx = np.flatnonzero(fp_mask)
                if not len(idx):
                    continue
                n_fp += len(idx)
                cat_n[patch_cat[st]] += len(idx)
                cl = cl_by_stem[st]
                for i in idx:
                    cat, pres = pixel_category(box_pixels(cl, r["preds"][i, :4]),
                                               float(r["max_iou"][i]))
                    pix_n[cat] += 1
                    for g, v in pres.items():
                        presence[g] += int(v)
            if sum(cat_n.values()) != n_fp or sum(pix_n.values()) != n_fp:
                raise SystemExit(f"[error] {name} {fp_def}: FP category totals inconsistent")
            for c in PATCH_CATS:
                share = cat_n[c] / n_fp if n_fp else float("nan")
                ref = n_patches_cat[c] / len(stems)
                fp_patch_share[(name, fp_def, c)] = share
                fp_patch_rows.append({"system": name, "fp_def": fp_def, "category": c,
                                      "n_fp": cat_n[c], "n_fp_total": n_fp,
                                      "share_fp": share, "n_patches": n_patches_cat[c],
                                      "share_patches": ref,
                                      "enrichment": share / ref if ref > 0 else ""})
            for c in PIXEL_CATS:
                share = pix_n[c] / n_fp if n_fp else float("nan")
                ref = float(pixel_ref[c])
                fp_pixel_share[(name, fp_def, c)] = share
                pres_key = "debris" if c in ("near_gt_iou_ge_0.1", "other_debris_pixels") \
                    else c
                fp_pixel_rows.append({
                    "system": name, "fp_def": fp_def, "category": c, "n_fp": pix_n[c],
                    "n_fp_total": n_fp, "share_fp": share, "pixel_share_reference": ref,
                    "enrichment_indicative": share / ref if ref > 0 else "",
                    "presence_group": pres_key if c != "unlabeled_only" else "",
                    "presence_rate": (presence[pres_key] / n_fp
                                      if c != "unlabeled_only" and n_fp else "")})
            print(f"      {name} {fp_def}: {n_fp} FP", flush=True)

    print("[7/8] missed-detection strata ...", flush=True)
    miss_rows = []
    recall_store = {}
    for name in system_names:
        per = res[name]
        for thr in IOU_THRS:
            for rdef in RECALL_DEFS:
                tallies = {}
                for st in stems:
                    gt_pred = per[st][f"gtp_{thr}"]
                    for gi, m in enumerate(gt_meta[st]):
                        j = gt_pred[gi]
                        hit = j >= 0 and (rdef == "max_recall" or bool(per[st]["sel"][j]))
                        for key in (("all", "all"), ("area", area_stratum(m["area"])),
                                    ("confidence", CONF_NAMES[m["conf"]])):
                            t = tallies.setdefault(key, [0, 0])
                            t[0] += 1
                            t[1] += int(hit)
                for (stype, stratum), (n, h) in sorted(tallies.items()):
                    recall_store[(name, thr, rdef, stype, stratum)] = (n, h)
                    miss_rows.append({"system": name, "iou_threshold": thr,
                                      "recall_def": rdef, "stratum_type": stype,
                                      "stratum": stratum, "n_gt": n, "n_recalled": h,
                                      "recall": h / n if n else ""})
                if tallies[("all", "all")][0] != 212:
                    raise SystemExit(f"[error] {name}: strata do not cover 212 GT")
    for name in system_names:
        n_max = recall_store[(name, 0.50, "max_recall", "all", "all")][1]
        n_tp = sum(int(res[name][st]["tp_0.5"].sum()) for st in stems)
        if n_max != n_tp:
            raise SystemExit(f"[error] {name}: max-recall hits {n_max} != TP {n_tp}")

    print("[8/8] aggregating over seeds and writing outputs ...", flush=True)

    def seed_agg(values):
        arr = np.asarray(values, float)
        return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))

    agg_fp_rows = []
    for fp_def in FP_DEFS:
        for level, cats, store in (("patch", PATCH_CATS, fp_patch_share),
                                   ("pixel", PIXEL_CATS, fp_pixel_share)):
            for c in cats:
                row = {"fp_def": fp_def, "level": level, "category": c}
                for ch in CHANNELS:
                    m, lo, hi = seed_agg([store[(f"{ch}_s{s}", fp_def, c)] for s in SEEDS])
                    row.update({f"{ch}_share_mean": m, f"{ch}_share_min": lo,
                                f"{ch}_share_max": hi})
                row["fdi_share"] = store[("fdi", fp_def, c)]
                agg_fp_rows.append(row)
    agg_miss_rows = []
    keys = sorted({k[1:] for k in recall_store}, key=lambda x: (x[0] != 0.5, x[1], x[2], x[3]))
    for thr, rdef, stype, stratum in keys:
        n = recall_store[("fdi", thr, rdef, stype, stratum)][0]
        row = {"iou_threshold": thr, "recall_def": rdef, "stratum_type": stype,
               "stratum": stratum, "n_gt": n}
        for ch in CHANNELS:
            vals = [recall_store[(f"{ch}_s{s}", thr, rdef, stype, stratum)][1] / n
                    for s in SEEDS]
            m, lo, hi = seed_agg(vals)
            row.update({f"{ch}_recall_mean": m, f"{ch}_recall_min": lo,
                        f"{ch}_recall_max": hi})
        row["fdi_recall"] = recall_store[("fdi", thr, rdef, stype, stratum)][1] / n
        agg_miss_rows.append(row)

    fb.write_csv(out / "operating_point_by_fold.csv", list(op_rows[0]), op_rows)
    fb.write_csv(out / "fp_patch_category.csv", list(fp_patch_rows[0]), fp_patch_rows)
    fb.write_csv(out / "fp_pixel_category.csv", list(fp_pixel_rows[0]), fp_pixel_rows)
    fb.write_csv(out / "fp_summary_over_seeds.csv", list(agg_fp_rows[0]), agg_fp_rows)
    fb.write_csv(out / "miss_strata.csv", list(miss_rows[0]), miss_rows)
    fb.write_csv(out / "miss_strata_over_seeds.csv", list(agg_miss_rows[0]), agg_miss_rows)

    gt_counts = {"area": {s: 0 for s in AREA_STRATA}, "confidence": {v: 0 for v in
                                                                     CONF_NAMES.values()}}
    for st in stems:
        for m in gt_meta[st]:
            gt_counts["area"][area_stratum(m["area"])] += 1
            gt_counts["confidence"][CONF_NAMES[m["conf"]]] += 1
    op_summary = {}
    for name in system_names:
        rows = [r for r in op_rows if r["system"] == name]
        n_sel = sum(r["n_selected"] for r in rows)
        n_tp = sum(r["n_tp_iou50"] for r in rows)
        op_summary[name] = {"n_selected": n_sel, "n_tp_iou50": n_tp,
                            "precision_iou50_pooled": n_tp / n_sel if n_sel else None}
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/v2_error_analysis.py",
        "script_sha256": fb.sha256_file(Path(__file__)),
        "inputs": {"fdi_oof_sha256": fb.sha256_file(fdi_dir / "fdi_oof.json"),
                   "patch_inventory_sha256": fb.sha256_file(cv.AUDIT_DIR / "patch_inventory.csv"),
                   "components_sha256": fb.sha256_file(
                       cv.AUDIT_DIR / "marine_debris_components.csv")},
        "protocol": {
            "systems": system_names,
            "yolo_predictions": "validated OOF, class 0 (debris) only, conf >= 0.001",
            "matching": "greedy per patch (= cv_pipeline.match_patch_tp); IoU 0.5 primary, "
                        "0.1 secondary; duplicates are FP",
            "top_ngt": "per CV fold, the n_gt(fold) highest-scoring test predictions",
            "fp_patch_level": "negative_category_min4 from patch_inventory.csv; reference = "
                              "share of patches",
            "fp_pixel_level": "FP box /2 to the 256 grid (floor/ceil, >=1 px); "
                              "near_gt if max IoU with a GT >= 0.1, else other_debris_pixels "
                              "if any class-1 pixel, else dominant annotated group "
                              "(algae_organic 2-4, ship 5, foam_waves_wakes 9/12/14, "
                              "cloud_shadow 6/13, water 7/8/10/11/15), else unlabeled_only; "
                              "reference = pixel share over all 1381 patches (indicative)",
            "miss_strata": "component area 4-7 / 8-15 / >=16 px; dominant confidence "
                           "1 high / 2 moderate / 3 low",
        },
        "checks": {
            "yolo_reference_AP_reproduced": True,
            "fdi_reference_AP_reproduced": True,
            "fdi_gt_identical_to_yolo_gt": True,
            "cl_tif_class_counts_identical_to_patch_inventory": True,
            "gt_linked_to_components": n_linked,
            "identical_box_ties": n_ambiguous,
            "matcher_identical_to_cv_pipeline_and_relaxed_match": True,
            "max_recall_hits_equal_tp": True,
        },
        "gt_counts": gt_counts,
        "pixel_reference_share": {k: float(v) for k, v in pixel_ref.items()},
        "operating_point_pooled": op_summary,
        "outputs": ["summary.json", "operating_point_by_fold.csv", "fp_patch_category.csv",
                    "fp_pixel_category.csv", "fp_summary_over_seeds.csv", "miss_strata.csv",
                    "miss_strata_over_seeds.csv"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
