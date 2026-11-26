"""Audit original Marine Debris mask agreement in locked same-acquisition pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs_mpb" / "mados_overlap"
PLAN = OUT / "analysis_plan.md"
PLAN_HASH = OUT / "analysis_plan.sha256"
MATCHES = OUT / "matches.csv"
RADIOMETRY = OUT / "radiometry.csv"
T2_SUMMARY = OUT / "t2_summary.json"
MADOS_RAW = ROOT / "data" / "MADOS"
MARIDA_RAW = ROOT / "data" / "raw" / "MARIDA" / "patches"
MADOS_ID = re.compile(r"^Scene_(\d+)_(\d+)$")
STRUCTURE = ndimage.generate_binary_structure(2, 1)
MIN_PIXELS = 4
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def locked_inputs() -> tuple[str, list[dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    plan_hash = sha256(PLAN)
    if plan_hash != PLAN_HASH.read_text(encoding="utf-8").split()[0].lower():
        raise RuntimeError("Analysis plan SHA-256 does not match its lock file")
    t2 = json.loads(T2_SUMMARY.read_text(encoding="utf-8"))
    if t2.get("analysis_plan_sha256") != plan_hash or t2.get("matches_sha256") != sha256(MATCHES):
        raise RuntimeError("T2 was not run with the locked plan and matches")
    if t2.get("stop_signal") is not False or t2.get("complete") is not True:
        raise RuntimeError("T2 stop signal or incomplete output forbids T3")
    matches = read_csv(MATCHES)
    radiometry = read_csv(RADIOMETRY)
    if len(matches) != t2.get("matches_rows") or len(radiometry) != t2.get("radiometry_rows"):
        raise RuntimeError("T1/T2 row counts changed after T2")
    rad_by_pair = {}
    for row in radiometry:
        key = row["mados_id"], row["marida_patch"]
        if key in rad_by_pair:
            raise RuntimeError(f"Duplicate T2 pair: {key}")
        rad_by_pair[key] = row
    if len({(row["mados_id"], row["marida_patch"]) for row in matches}) != len(matches):
        raise RuntimeError("Duplicate T1 pair")
    if any((row["mados_id"], row["marida_patch"]) not in rad_by_pair for row in matches):
        raise RuntimeError("T1/T2 pair sets differ")
    return plan_hash, matches, rad_by_pair


def load_mados_mask(stem: str) -> np.ndarray:
    match = MADOS_ID.fullmatch(stem)
    if match is None:
        raise ValueError(f"Invalid MADOS id: {stem}")
    scene, crop = match.groups()
    path = MADOS_RAW / f"Scene_{scene}" / "10" / f"Scene_{scene}_L2R_cl_{crop}.tif"
    with rasterio.open(path) as source:
        mask = source.read(1) == 1
    if mask.shape != (240, 240):
        raise ValueError(f"Unexpected MADOS mask shape: {path}: {mask.shape}")
    return mask


def load_marida_mask(stem: str) -> np.ndarray:
    path = MARIDA_RAW / stem.rsplit("_", 1)[0] / f"{stem}_cl.tif"
    with rasterio.open(path) as source:
        mask = source.read(1) == 1
    if mask.shape != (256, 256):
        raise ValueError(f"Unexpected MARIDA mask shape: {path}: {mask.shape}")
    return mask


def components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labeled, count = ndimage.label(mask, structure=STRUCTURE)
    sizes = np.bincount(labeled.ravel(), minlength=count + 1)
    eligible = sizes >= MIN_PIXELS
    eligible[0] = False
    return labeled, eligible


def window(dy: int, dx: int) -> tuple[slice, slice, slice, slice]:
    r0, c0 = max(0, -dy), max(0, -dx)
    r1, c1 = min(240, 256 - dy), min(240, 256 - dx)
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"Empty aligned window for dy={dy}, dx={dx}")
    return slice(r0, r1), slice(c0, c1), slice(r0 + dy, r1 + dy), slice(c0 + dx, c1 + dx)


def bool_true(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"Unexpected T2 Boolean: {value!r}")
    return value == "True"


def tally(a: np.ndarray, b: np.ndarray, selected: np.ndarray) -> dict[str, int]:
    return {
        "px_mados": int((a & selected).sum()),
        "px_marida": int((b & selected).sum()),
        "px_intersection": int((a & b & selected).sum()),
        "px_union": int(((a | b) & selected).sum()),
        "px_mados_only": int((a & ~b & selected).sum()),
        "px_marida_only": int((~a & b & selected).sum()),
    }


def component_ids(labels: np.ndarray, eligible: np.ndarray, selected: np.ndarray) -> np.ndarray:
    ids = np.unique(labels[selected])
    return ids[eligible[ids]]


def audit() -> None:
    plan_hash, matches, rad_by_pair = locked_inputs()
    selected = []
    excluded = Counter()
    for row in matches:
        rad = rad_by_pair[row["mados_id"], row["marida_patch"]]
        if row["marida_split"] != rad["marida_split"]:
            raise RuntimeError(f"T1/T2 split conflict: {row['marida_patch']}")
        if row["status"] != "valid" or row["dy"] == "" or row["dx"] == "":
            excluded["alignment_unresolved"] += 1
        elif rad["status"] != "ok":
            excluded[rad["status"]] += 1
        elif bool_true(rad["same_acq_same_processing"]):
            selected.append(row)
        else:
            excluded["not_supported_same_acq_same_processing"] += 1
    selected.sort(key=lambda row: (row["mados_id"], -float(row["match_score"]), row["marida_patch"]))
    mados_cache = {}
    marida_cache = {}
    owned = {}
    mados_components = {}
    marida_components = {}
    rows = []
    sums = defaultdict(Counter)
    ambiguous_mados_component_splits = 0
    for row in selected:
        mados_id, marida_patch, split = row["mados_id"], row["marida_patch"], row["marida_split"]
        if split not in SPLITS:
            raise ValueError(f"Unexpected MARIDA split: {split}")
        if mados_id not in mados_cache:
            mask = load_mados_mask(mados_id)
            mados_cache[mados_id] = (mask, *components(mask))
            owned[mados_id] = np.zeros(mask.shape, dtype=bool)
        if marida_patch not in marida_cache:
            mask = load_marida_mask(marida_patch)
            marida_cache[marida_patch] = (mask, *components(mask))
        m_mask, m_labels, m_eligible = mados_cache[mados_id]
        r_mask, r_labels, r_eligible = marida_cache[marida_patch]
        ar, ac, br, bc = window(int(row["dy"]), int(row["dx"]))
        a, b = m_mask[ar, ac], r_mask[br, bc]
        available = ~owned[mados_id][ar, ac]
        owned[mados_id][ar, ac] = True
        raw = tally(a, b, np.ones(a.shape, dtype=bool))
        assigned = tally(a, b, available)
        sums["all"].update(assigned)
        sums[split].update(assigned)
        sums["all"]["pairs"] += 1
        sums[split]["pairs"] += 1
        m_window_labels = m_labels[ar, ac]
        r_window_labels = r_labels[br, bc]
        m_ids = component_ids(m_window_labels, m_eligible, available)
        r_ids = component_ids(r_window_labels, r_eligible, available)
        pair_m_hits = 0
        pair_r_hits = 0
        for component_id in m_ids:
            key = (mados_id, int(component_id))
            hit = bool(((m_window_labels == component_id) & b & available).any())
            pair_m_hits += int(hit)
            existing = mados_components.get(key)
            if existing is None:
                mados_components[key] = {"split": split, "hit": hit}
            else:
                ambiguous_mados_component_splits += int(existing["split"] != split)
                existing["hit"] |= hit
        for component_id in r_ids:
            key = (marida_patch, int(component_id))
            hit = bool(((r_window_labels == component_id) & a & available).any())
            pair_r_hits += int(hit)
            existing = marida_components.get(key)
            if existing is None:
                marida_components[key] = {"split": split, "hit": hit}
            else:
                existing["hit"] |= hit
        record = {"mados_id": mados_id, "marida_patch": marida_patch, "marida_split": split,
                  "dy": int(row["dy"]), "dx": int(row["dx"]), "match_score": row["match_score"],
                  "n_window_px": int(a.size), "n_assigned_px": int(available.sum()),
                  "n_duplicate_mados_px": int(a.size - available.sum())}
        record.update({f"raw_{key}": value for key, value in raw.items()})
        record.update({f"assigned_{key}": value for key, value in assigned.items()})
        record.update(assigned)
        record["px_iou"] = assigned["px_intersection"] / assigned["px_union"] if assigned["px_union"] else ""
        record["comp_mados"] = int(len(m_ids))
        record["comp_mados_in_marida"] = pair_m_hits
        record["comp_marida"] = int(len(r_ids))
        record["comp_marida_in_mados"] = pair_r_hits
        record["raw_px_iou"] = raw["px_intersection"] / raw["px_union"] if raw["px_union"] else ""
        record["assigned_px_iou"] = assigned["px_intersection"] / assigned["px_union"] if assigned["px_union"] else ""
        rows.append(record)
    for component in mados_components.values():
        for group in ("all", component["split"]):
            sums[group]["comp_mados"] += 1
            sums[group]["comp_mados_in_marida"] += int(component["hit"])
    for component in marida_components.values():
        for group in ("all", component["split"]):
            sums[group]["comp_marida"] += 1
            sums[group]["comp_marida_in_mados"] += int(component["hit"])
    pair_fields = ["mados_id", "marida_patch", "marida_split", "dy", "dx", "match_score",
                   "n_window_px", "n_assigned_px", "n_duplicate_mados_px"]
    measures = ("px_mados", "px_marida", "px_intersection", "px_union", "px_mados_only", "px_marida_only")
    pair_fields += [f"raw_{name}" for name in measures] + [f"assigned_{name}" for name in measures]
    pair_fields += [*measures, "px_iou", "comp_mados", "comp_mados_in_marida",
                    "comp_marida", "comp_marida_in_mados", "raw_px_iou", "assigned_px_iou"]
    with (OUT / "label_agreement.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    for group in ("all", *SPLITS):
        counts = sums[group]
        record = {"marida_split": group}
        record.update({name: counts[name] for name in (*measures, "pairs", "comp_mados",
                                                 "comp_mados_in_marida", "comp_marida", "comp_marida_in_mados")})
        record["px_iou_pooled"] = counts["px_intersection"] / counts["px_union"] if counts["px_union"] else ""
        record["comp_mados_coverage"] = (counts["comp_mados_in_marida"] / counts["comp_mados"]
                                           if counts["comp_mados"] else "")
        record["comp_marida_coverage"] = (counts["comp_marida_in_mados"] / counts["comp_marida"]
                                           if counts["comp_marida"] else "")
        record["comp_mados_only"] = counts["comp_mados"] - counts["comp_mados_in_marida"]
        record["comp_marida_only"] = counts["comp_marida"] - counts["comp_marida_in_mados"]
        summary_rows.append(record)
    with (OUT / "label_agreement_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest = {"algorithm": "mados_overlap_t3_v1", "analysis_plan_sha256": plan_hash,
                "matches_sha256": sha256(MATCHES), "radiometry_sha256": sha256(RADIOMETRY),
                "t2_summary_sha256": sha256(T2_SUMMARY), "script_sha256": sha256(Path(__file__)),
                "matches_rows": len(matches), "eligible_pairs": len(selected), "excluded_pairs": dict(excluded),
                "mados_crops": len(mados_cache), "marida_patches": len(marida_cache),
                "ambiguous_mados_component_splits": ambiguous_mados_component_splits,
                "pair_table_sha256": sha256(OUT / "label_agreement.csv"),
                "summary_table_sha256": sha256(OUT / "label_agreement_summary.csv")}
    (OUT / "t3_summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest, "pooled": summary_rows[0]}, ensure_ascii=False))


def self_test() -> None:
    assert window(20, 22) == (slice(0, 236), slice(0, 234), slice(20, 256), slice(22, 256))
    a = np.array([[True, False], [True, False]])
    b = np.array([[True, True], [False, False]])
    assert tally(a, b, np.ones((2, 2), dtype=bool)) == {
        "px_mados": 2, "px_marida": 2, "px_intersection": 1,
        "px_union": 3, "px_mados_only": 1, "px_marida_only": 1}
    labeled, eligible = components(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool))
    assert component_ids(labeled, eligible, np.ones((3, 3), dtype=bool)).tolist() == [1]
    print("T3 self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else audit()
