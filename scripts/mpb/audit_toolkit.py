"""
audit_toolkit.py -- reviewer-side helper functions for the MARIDA detection audit.

These are TEMPLATES written without access to your repository: adapt names,
column layouts and band orders to your own files. Every function has a small
self-test at the bottom (run:  python audit_toolkit.py).

Contents
  1. ap_from_ledger        101-point AP from a per-prediction TP ledger
  2. scene_bootstrap       scene-clustered bootstrap of an AP difference
  3. paired_scene_swap     exact/Monte-Carlo paired permutation test (scene level)
  4. scene_jackknife       leave-one-scene-out influence on an AP difference
  5. ci_vs_margin          is the CI inside / outside an equivalence margin?
  6. grouped_scene_folds   balanced leave-scene-out folds (debris counts balanced)
  7. parse_marida_name / scene_overlap   MARIDA vs external-set overlap check
  8. cohens_kappa          agreement for manual review of the 212 candidates
  9. spectral_indices      NDVI / NDWI / FDI / PI from an 11-band MARIDA array
 10. rot90_yolo_boxes      lossless 90-degree augmentation for YOLO boxes
"""
from __future__ import annotations

import itertools
import re
from collections import defaultdict
from datetime import date

import numpy as np

# --------------------------------------------------------------------------
# 1. AP from a ledger
# --------------------------------------------------------------------------
# Ledger convention: one entry per retained prediction of the target class,
#   scene id, confidence score, tp flag (1 = matched at IoU>=0.5, else 0);
# plus n_gt per scene (number of ground-truth boxes in that scene).


def ap_from_ledger(scores, tp, n_gt, n_points: int = 101) -> float:
    """101-point interpolated AP (COCO-style precision envelope)."""
    scores = np.asarray(scores, float)
    tp = np.asarray(tp, float)
    if n_gt == 0:
        return float("nan")
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    # precision envelope (monotone non-increasing from the right)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, n_points)
    idx = np.searchsorted(recall, grid, side="left")
    vals = np.where(idx < precision.size, precision[np.minimum(idx, precision.size - 1)], 0.0)
    return float(vals.mean())


def _pool(ledger: dict, scene_list):
    """ledger: scene -> (scores, tp, n_gt). scene_list may contain repeats."""
    s, t, n = [], [], 0
    for sc in scene_list:
        sc_s, sc_t, sc_n = ledger[sc]
        s.append(sc_s)
        t.append(sc_t)
        n += sc_n
    if not s:
        return np.array([]), np.array([]), 0
    return np.concatenate(s), np.concatenate(t), n


def pooled_ap(ledger: dict, scene_list) -> float:
    s, t, n = _pool(ledger, scene_list)
    return ap_from_ledger(s, t, n)


# --------------------------------------------------------------------------
# 2. Scene-clustered bootstrap of a paired AP difference (one seed or averaged)
# --------------------------------------------------------------------------


def scene_bootstrap(ledgers_a: list[dict], ledgers_b: list[dict], B: int = 10000,
                    seed: int = 0, alpha: float = 0.05):
    """
    ledgers_a / ledgers_b: lists (one dict per training seed) of
        scene -> (scores, tp, n_gt)    for model A and model B on the SAME scenes.
    Statistic: mean over seeds of [AP_a - AP_b] on the resampled scene set.
    Returns (point, (lo, hi), draws).
    """
    scenes = sorted(ledgers_a[0].keys())
    rng = np.random.default_rng(seed)

    def stat(scene_list):
        d = []
        for la, lb in zip(ledgers_a, ledgers_b):
            a, b = pooled_ap(la, scene_list), pooled_ap(lb, scene_list)
            if not (np.isnan(a) or np.isnan(b)):
                d.append(a - b)
        return np.mean(d) if d else np.nan

    point = stat(scenes)
    draws = np.array([stat(rng.choice(scenes, size=len(scenes), replace=True)) for _ in range(B)])
    draws = draws[~np.isnan(draws)]
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, (float(lo), float(hi)), draws


# --------------------------------------------------------------------------
# 3. Paired scene-swap permutation test
# --------------------------------------------------------------------------


def paired_scene_swap(ledgers_a: list[dict], ledgers_b: list[dict], n_mc: int = 20000,
                      seed: int = 0, exact_max_scenes: int = 14):
    """
    H0: models A and B are exchangeable within every scene. Randomly swap the
    two models' ledgers scene by scene, recompute mean-over-seeds AP difference.
    Two-sided p. Exact if #scenes <= exact_max_scenes (2^k swaps), else Monte Carlo.
    """
    scenes = sorted(ledgers_a[0].keys())
    k = len(scenes)

    def stat(mask):
        diffs = []
        for la, lb in zip(ledgers_a, ledgers_b):
            A = {sc: (lb[sc] if m else la[sc]) for sc, m in zip(scenes, mask)}
            Bd = {sc: (la[sc] if m else lb[sc]) for sc, m in zip(scenes, mask)}
            diffs.append(pooled_ap(A, scenes) - pooled_ap(Bd, scenes))
        return float(np.mean(diffs))

    obs = stat([0] * k)
    if k <= exact_max_scenes:
        masks = itertools.product([0, 1], repeat=k)
        null = np.array([stat(m) for m in masks])
    else:
        rng = np.random.default_rng(seed)
        null = np.array([stat(rng.integers(0, 2, k)) for _ in range(n_mc)])
    extreme = int(np.count_nonzero(np.abs(null) >= abs(obs) - 1e-12))
    # Exact enumeration includes the observed allocation; Monte Carlo uses
    # the plus-one correction so a sampled permutation p-value is never zero.
    p = extreme / len(null) if k <= exact_max_scenes else (extreme + 1) / (n_mc + 1)
    return obs, p, null


# --------------------------------------------------------------------------
# 4. Leave-one-scene-out influence
# --------------------------------------------------------------------------


def scene_jackknife(ledgers_a: list[dict], ledgers_b: list[dict]):
    """Return {scene: AP-difference when that scene is dropped} and the full value."""
    scenes = sorted(ledgers_a[0].keys())

    def diff(sl):
        return float(np.mean([pooled_ap(la, sl) - pooled_ap(lb, sl)
                              for la, lb in zip(ledgers_a, ledgers_b)]))

    full = diff(scenes)
    return full, {sc: diff([s for s in scenes if s != sc]) for sc in scenes}


# --------------------------------------------------------------------------
# 5. Equivalence-style reading of a CI
# --------------------------------------------------------------------------


def ci_vs_margin(lo: float, hi: float, margin: float) -> str:
    """Read an interval against a margin; this is not a formal TOST procedure."""
    if -margin <= lo and hi <= margin:
        return f"CI contained within +/-{margin} (descriptive; not a TOST test)"
    if lo > margin:
        return f"larger than +{margin}"
    if hi < -margin:
        return f"smaller than -{margin}"
    if lo > 0 or hi < 0:
        return "non-zero but not clearly beyond the margin"
    return "inconclusive (CI includes both 0 and values beyond the margin)"


# --------------------------------------------------------------------------
# 6. Balanced leave-scene-out folds
# --------------------------------------------------------------------------


def grouped_scene_folds(scene_counts: dict, k: int = 9, seed: int = 0):
    """
    scene_counts: scene -> number of debris instances (can be 0).
    Greedy balancing of debris counts across k folds; scenes are never split.
    Returns list of k lists of scenes. Use fold i as test, fold (i+1)%k as
    validation (threshold/checkpoint selection), the rest as training.
    """
    rng = np.random.default_rng(seed)
    scenes = list(scene_counts)
    rng.shuffle(scenes)
    scenes.sort(key=lambda s: -scene_counts[s])  # stable: ties stay shuffled
    folds = [[] for _ in range(k)]
    load = [0] * k
    size = [0] * k
    for sc in scenes:
        j = min(range(k), key=lambda i: (load[i], size[i]))
        folds[j].append(sc)
        load[j] += scene_counts[sc]
        size[j] += 1
    return folds


# --------------------------------------------------------------------------
# 7. Overlap check (tile + date), MARIDA patch names
# --------------------------------------------------------------------------

_NAME = re.compile(r"^S2_(\d{1,2})-(\d{1,2})-(\d{2})_(\w{5})(?:_(\d+))?$")


def parse_marida_name(name: str):
    """'S2_14-3-20_18QYF_3' -> dict(date=2020-03-14, tile='18QYF', patch=3)."""
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"unrecognised MARIDA name: {name}")
    d, mo, yy, tile, patch = m.groups()
    return {"date": date(2000 + int(yy), int(mo), int(d)), "tile": tile,
            "patch": None if patch is None else int(patch)}


def scene_overlap(a: list[dict], b: list[dict], day_tol: int = 0):
    """
    a, b: lists of dict(tile=..., date=datetime.date, id=...) -- one per scene.
    Returns pairs from a x b with the same MGRS tile and |date diff| <= day_tol.
    Same tile + same date = same Sentinel-2 acquisition. Same tile, different
    date = same geography (not independent in location); report both.
    """
    by_tile = defaultdict(list)
    for r in b:
        by_tile[r["tile"]].append(r)
    out = []
    for r in a:
        for q in by_tile.get(r["tile"], []):
            dd = abs((r["date"] - q["date"]).days)
            if dd <= day_tol:
                out.append((r["id"], q["id"], dd))
    return out


# --------------------------------------------------------------------------
# 8. Agreement statistics for the manual review of candidate boxes
# --------------------------------------------------------------------------


def cohens_kappa(r1, r2):
    r1, r2 = np.asarray(r1), np.asarray(r2)
    cats = np.unique(np.concatenate([r1, r2]))
    po = float((r1 == r2).mean())
    pe = float(sum((r1 == c).mean() * (r2 == c).mean() for c in cats))
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


# --------------------------------------------------------------------------
# 9. Spectral indices from the 11-band MARIDA stack
# --------------------------------------------------------------------------
# VERIFY the band order of YOUR files (the GeoTIFF metadata / dataset docs).
# The order below is the usual MARIDA patch order; change it if yours differs.
BAND_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
WL = {"B4": 665.0, "B6": 740.0, "B8": 832.8, "B11": 1613.7}  # nm, Sentinel-2A nominal


def spectral_indices(x: np.ndarray, band_order=BAND_ORDER, eps: float = 1e-6):
    """x: (C,H,W) reflectance. Returns dict of (H,W) arrays."""
    b = {n: x[i].astype(np.float64) for i, n in enumerate(band_order)}
    ndvi = (b["B8"] - b["B4"]) / (b["B8"] + b["B4"] + eps)
    ndwi = (b["B3"] - b["B8"]) / (b["B3"] + b["B8"] + eps)
    # Floating Debris Index (Biermann et al., 2020): needs the red-edge band B6
    slope = (WL["B8"] - WL["B4"]) / (WL["B11"] - WL["B4"]) * 10.0
    fdi = b["B8"] - (b["B6"] + (b["B11"] - b["B6"]) * slope)
    pi = b["B8"] / (b["B8"] + b["B4"] + eps)  # plastic index
    return {"NDVI": ndvi, "NDWI": ndwi, "FDI": fdi, "PI": pi}


# --------------------------------------------------------------------------
# 10. Lossless 90-degree augmentation for YOLO boxes (square images)
# --------------------------------------------------------------------------


def rot90_yolo_boxes(boxes: np.ndarray, k: int = 1) -> np.ndarray:
    """
    boxes: (N,4) normalised [xc, yc, w, h]; image rotated with np.rot90(img, k)
    (counter-clockwise, rows=y, cols=x). Exact for axis-aligned boxes, so tiny
    objects are not inflated the way arbitrary-angle rotation inflates them.
    """
    b = boxes.copy().astype(float)
    for _ in range(k % 4):
        xc, yc, w, h = b[:, 0].copy(), b[:, 1].copy(), b[:, 2].copy(), b[:, 3].copy()
        b[:, 0], b[:, 1], b[:, 2], b[:, 3] = yc, 1.0 - xc, h, w
    return b


# ==========================================================================
# Self-tests on synthetic data
# ==========================================================================


def _synthetic_ledger(rng, scenes, quality):
    led = {}
    for sc in scenes:
        n_gt = int(rng.integers(0, 6))
        n_pred = int(rng.integers(3, 12))
        scores = rng.random(n_pred)
        # higher 'quality' => true positives get higher scores
        tp = (rng.random(n_pred) < scores * quality).astype(float)
        led[sc] = (scores, np.minimum(tp, 1.0), n_gt)
    return led


def _selftest():
    rng = np.random.default_rng(1)

    # AP sanity: perfect ranking -> 1; empty predictions -> 0
    assert abs(ap_from_ledger([0.9, 0.8, 0.7], [1, 1, 1], 3) - 1.0) < 1e-9
    assert ap_from_ledger([], [], 3) == 0.0
    # one TP found at rank 2 of 2, n_gt=1 -> precision 0.5 at all recall points
    assert abs(ap_from_ledger([0.9, 0.8], [0, 1], 1) - 0.5) < 1e-9

    scenes = [f"S{i}" for i in range(10)]
    A = [_synthetic_ledger(rng, scenes, 0.9) for _ in range(3)]
    B = [_synthetic_ledger(rng, scenes, 0.5) for _ in range(3)]
    pt, ci, _ = scene_bootstrap(A, B, B=300, seed=0)
    assert ci[0] < ci[1]
    obs, p, null = paired_scene_swap(A, B)
    assert null.size == 2 ** 10 and 0 < p <= 1
    # identical models -> observed difference exactly 0, p = 1
    obs0, p0, _ = paired_scene_swap(A, A)
    assert obs0 == 0 and p0 == 1.0
    full, jk = scene_jackknife(A, B)
    assert set(jk) == set(scenes)
    assert ci_vs_margin(-0.202, 0.019, 0.05).startswith("inconclusive")
    assert ci_vs_margin(-0.03, 0.04, 0.05).startswith("CI contained")

    folds = grouped_scene_folds({f"S{i}": int(rng.integers(0, 20)) for i in range(63)}, k=9)
    assert sum(len(f) for f in folds) == 63 and len(set(itertools.chain(*folds))) == 63

    r = parse_marida_name("S2_14-3-20_18QYF_3")
    assert r["date"] == date(2020, 3, 14) and r["tile"] == "18QYF" and r["patch"] == 3
    ov = scene_overlap([{"id": "m1", "tile": "18QYF", "date": date(2020, 3, 14)}],
                       [{"id": "d1", "tile": "18QYF", "date": date(2020, 3, 14)},
                        {"id": "d2", "tile": "16PCC", "date": date(2018, 2, 26)}])
    assert ov == [("m1", "d1", 0)]

    po, kappa = cohens_kappa([1, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1])
    assert abs(po - 4 / 6) < 1e-9 and -1 <= kappa <= 1

    x = rng.random((11, 8, 8)) * 0.1
    idx = spectral_indices(x)
    assert all(v.shape == (8, 8) for v in idx.values())

    # rot90 box transform vs. brute force on a mask
    N = 32
    mask = np.zeros((N, N), bool)
    mask[5:9, 12:20] = True                      # rows 5..8, cols 12..19
    ys, xs = np.where(mask)
    box = np.array([[(xs.min() + xs.max() + 1) / 2 / N, (ys.min() + ys.max() + 1) / 2 / N,
                     (xs.max() - xs.min() + 1) / N, (ys.max() - ys.min() + 1) / N]])
    for k in range(4):
        m2 = np.rot90(mask, k)
        ys2, xs2 = np.where(m2)
        ref = np.array([[(xs2.min() + xs2.max() + 1) / 2 / N, (ys2.min() + ys2.max() + 1) / 2 / N,
                         (xs2.max() - xs2.min() + 1) / N, (ys2.max() - ys2.min() + 1) / N]])
        assert np.allclose(rot90_yolo_boxes(box, k), ref), k
    print("all self-tests passed")


if __name__ == "__main__":
    _selftest()
