"""Build the locked full MADOS–MARIDA duplicate map and source-specific crosstab."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.mpb import mados_overlap_t1 as t1
from scripts.mpb import mados_overlap_t2 as t2
from scripts.mpb import mados_overlap_t3 as t3
from scripts.mpb import match_mados_marida_imagery as matcher_code


OUT = ROOT / "runs_mpb" / "mados_overlap" / "v2"
PLAN = OUT.parent / "analysis_plan_v2.md"
PLAN_LOCK = OUT.parent / "analysis_plan_v2.sha256"
OLD_MATCHES = OUT.parent / "matches.csv"
SPLITS = ("test", "train", "val")
CANDIDATES = 160
NEGATIVE_SEED = 20261015
NEGATIVE_PER_CROP = 20
RANK_CUTOFFS = (40, 80, 160)
SIFT_FEATURES = 600
SCORE_THRESHOLD = 0.95


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_plan() -> str:
    actual = sha256(PLAN)
    expected = PLAN_LOCK.read_text(encoding="utf-8").split()[0].lower()
    if actual != expected:
        raise RuntimeError(f"Locked T6/T7 plan changed: {actual} != {expected}")
    return actual


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def official_splits() -> dict[str, list[str]]:
    splits = {}
    seen = set()
    for split, expected in (("train", 1433), ("val", 642), ("test", 728)):
        path = t2.MADOS_RAW / "splits" / f"{split}_X.txt"
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(ids) != expected or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise RuntimeError(f"Unexpected official MADOS {split} split")
        seen.update(ids)
        splits[split] = ids
    if len(seen) != 2803:
        raise RuntimeError("Official MADOS split union must contain 2803 crops")
    return splits


def marida_patches() -> list[tuple[Path, str, str]]:
    _, _, patches = matcher_code.find_sources()
    if len(patches) != 1381:
        raise RuntimeError(f"Expected 1381 MARIDA patches, got {len(patches)}")
    return patches


def old_pairs() -> set[tuple[str, str]]:
    rows = read_csv(OLD_MATCHES)
    pairs = {(row["mados_id"], row["marida_patch"]) for row in rows if row["status"] == "valid"}
    if len(pairs) != 342:
        raise RuntimeError(f"Expected 342 valid old pairs, got {len(pairs)}")
    return pairs


def score_pair(crop: str, target_index: int, query_features: tuple,
               target_features: list[tuple], patches: list[tuple],
               source_bands: dict, target_bands: dict) -> dict:
    query_shape, query_kp, query_desc = query_features
    target_shape, target_kp, target_desc = target_features[target_index]
    fit = t1.fit_pair(query_kp, query_desc, target_kp, target_desc, query_shape, target_shape)
    b3_corr = float("nan")
    b2_corr = float("nan")
    warped_overlap = 0
    score = 0.0
    patch_path, scene, split = patches[target_index]
    if fit["geometry_pass"]:
        if crop not in source_bands:
            source_bands[crop], _ = matcher_code.load_mados(crop, ("B3", "B2"))
        if target_index not in target_bands:
            target_bands[target_index] = matcher_code.load_marida(patch_path, ("B3", "B2"))
        b3_corr, warped_overlap = matcher_code.band_correlation(
            source_bands[crop]["B3"], target_bands[target_index]["B3"], fit["homography"])
        b2_corr, _ = matcher_code.band_correlation(
            source_bands[crop]["B2"], target_bands[target_index]["B2"], fit["homography"])
        if warped_overlap >= matcher_code.MIN_OVERLAP_PIXELS and np.isfinite(b3_corr) and np.isfinite(b2_corr):
            score = min(b3_corr, b2_corr)
    fields = {key: t1.safe_number(value) for key, value in fit.items() if key != "homography"}
    return {"mados_id": crop, "marida_patch": patch_path.stem, "marida_scene": scene,
            "marida_split": split, "marida_tile": scene.rsplit("_", 1)[-1],
            "score": round(float(score), 6), "b3_correlation": t1.safe_number(b3_corr),
            "b2_correlation": t1.safe_number(b2_corr), "warped_overlap_px": warped_overlap,
            "confirmed": bool(score >= SCORE_THRESHOLD and fit["integer_pass"]), **fields}


def scan() -> None:
    locked_plan()
    cv2.setRNGSeed(0)
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    splits = official_splits()
    patches = marida_patches()
    old = old_pairs()
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    target_features = []
    descriptors = []
    for number, (path, _, _) in enumerate(patches, 1):
        image = matcher_code.load_marida(path)["B4"]
        kp, desc = sift.detectAndCompute(image, None)
        target_features.append((image.shape, kp, desc))
        descriptors.append(desc if desc is not None and len(desc) >= 2 else np.zeros((1, 128), np.float32))
        if number % 200 == 0:
            print(f"Indexed MARIDA {number}/{len(patches)}", flush=True)
    index = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=96))
    index.add(descriptors)
    index.train()
    progress = OUT / "scan_progress.jsonl"
    processed = set()
    prior_rows = []
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["mados_id"] in processed:
                raise RuntimeError("Duplicate crop in scan progress")
            processed.add(record["mados_id"])
            prior_rows.append(record)
    expected_ids = {crop for ids in splits.values() for crop in ids}
    if not processed.issubset(expected_ids):
        raise RuntimeError("Foreign crop in scan progress")
    records = prior_rows
    with progress.open("a", encoding="utf-8") as handle:
        for split in SPLITS:
            for number, crop in enumerate(splits[split], 1):
                if crop in processed:
                    continue
                images, _ = matcher_code.load_mados(crop)
                image = images["B4"]
                kp, desc = sift.detectAndCompute(image, None)
                query = (image.shape, kp, desc)
                candidates = matcher_code.retrieve_candidates(desc, index, CANDIDATES) if desc is not None else []
                source_bands = {}
                target_bands = {}
                confirmed = []
                for rank, (candidate_index, hits) in enumerate(candidates, 1):
                    result = score_pair(crop, candidate_index, query, target_features, patches,
                                        source_bands, target_bands)
                    if result["confirmed"]:
                        result["mados_split"] = split
                        result["retrieval_rank"] = rank
                        result["retrieval_hits"] = hits
                        confirmed.append(result)
                record = {"mados_id": crop, "mados_split": split,
                          "candidates_checked": len(candidates), "confirmed_pairs": confirmed}
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                records.append(record)
                processed.add(crop)
                if number % 25 == 0:
                    print(f"Scanned {split} {number}/{len(splits[split])}; confirmed crops "
                          f"{sum(bool(row['confirmed_pairs']) for row in records)}", flush=True)
            if split == "test":
                recovered = {(row["mados_id"], pair["marida_patch"])
                             for row in records if row["mados_split"] == "test" for pair in row["confirmed_pairs"]}
                missing = old - recovered
                ranks = [pair["retrieval_rank"] for row in records for pair in row["confirmed_pairs"]
                         if (row["mados_id"], pair["marida_patch"]) in old]
                coverage = {str(cutoff): sum(rank <= cutoff for rank in ranks) for cutoff in RANK_CUTOFFS}
                (OUT / "old_pair_recovery.json").write_text(json.dumps({
                    "expected_pairs": len(old), "recovered_pairs": len(old & recovered),
                    "missing": sorted(map(list, missing)), "rank_coverage": coverage,
                    "plan_sha256": locked_plan()}, indent=2) + "\n", encoding="utf-8")
                print(f"Old pair recovery: {len(old & recovered)}/{len(old)}; ranks {coverage}", flush=True)
                if missing:
                    raise RuntimeError(f"Stopped: {len(missing)} old pairs absent after top-160 test scan")
    if len(records) != 2803:
        raise RuntimeError(f"Expected 2803 scan records, got {len(records)}")
    pairs = [pair for row in records for pair in row["confirmed_pairs"]]
    keys = [(pair["mados_id"], pair["marida_patch"]) for pair in pairs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate pair in scan records")
    write_csv(OUT / "scan_pairs.csv", pairs, sorted({key for row in pairs for key in row}))
    write_csv(OUT / "unconfirmed_crops.csv",
              [{"mados_id": row["mados_id"], "mados_split": row["mados_split"],
                "candidates_checked": row["candidates_checked"], "status": "unconfirmed_in_top160"}
               for row in records if not row["confirmed_pairs"]],
              ["mados_id", "mados_split", "candidates_checked", "status"])
    print(f"Scan complete: {len(pairs)} pairs, {len(records)} crops", flush=True)


def radiometry_and_labels(pair: dict, image_cache: dict, mask_cache: dict) -> dict:
    crop, patch = pair["mados_id"], pair["marida_patch"]
    dy, dx = int(pair["dy"]), int(pair["dx"])
    slices = t2.window(dy, dx)
    if slices is None:
        raise RuntimeError(f"Empty integer window: {crop}, {patch}")
    ar, ac, br, bc = slices
    if crop not in image_cache:
        image_cache[crop] = t2.load_mados_6b(crop)
    source = image_cache[crop]
    target = t2.load_marida_6b(patch)
    a, b = source[:, ar, ac], target[:, br, bc]
    denominator = b[1] + b[3]
    valid = np.isfinite(a).all(0) & np.isfinite(b).all(0) & (np.abs(denominator) > 1e-6)
    ndwi = np.divide(b[1] - b[3], denominator,
                     out=np.full_like(denominator, np.nan), where=np.abs(denominator) > 1e-6)
    valid &= ndwi > 0
    valid = ndimage.binary_erosion(valid, iterations=3, border_value=0)
    n_water = int(valid.sum())
    record = {"n_water_px": n_water, "same_acq": "undetermined" if n_water < 200 else "False"}
    if n_water >= 200:
        for index, band in enumerate(t2.BANDS):
            corr, slope, intercept, ratio = t2.fit(b[index][valid], a[index][valid])
            record.update({f"{band}_r": corr, f"{band}_slope": slope,
                           f"{band}_icpt": intercept, f"{band}_medratio": ratio})
        same = all(np.isfinite(record[f"{band}_r"]) and record[f"{band}_r"] > 0.98
                   and np.isfinite(record[f"{band}_slope"])
                   and abs(record[f"{band}_slope"] - 1) <= 0.05 for band in t2.BANDS[:4])
        record["same_acq"] = str(same)
    if crop not in mask_cache:
        mask_cache[crop] = t3.load_mados_mask(crop)
    full_ma = mask_cache[crop]
    full_mb = t3.load_marida_mask(patch)
    ma = full_ma[ar, ac]
    mb = full_mb[br, bc]
    record.update(t3.tally(ma, mb, np.ones(ma.shape, dtype=bool)))
    full_la, ea = t3.components(full_ma)
    full_lb, eb = t3.components(full_mb)
    la, lb = full_la[ar, ac], full_lb[br, bc]
    ida = t3.component_ids(la, ea, np.ones(ma.shape, dtype=bool))
    idb = t3.component_ids(lb, eb, np.ones(mb.shape, dtype=bool))
    record.update({"comp_mados": int(len(ida)), "comp_marida": int(len(idb)),
                   "comp_mados_in_marida": sum(bool(np.any((la == item) & mb)) for item in ida),
                   "comp_marida_in_mados": sum(bool(np.any((lb == item) & ma)) for item in idb)})
    return record


def analyze() -> None:
    locked_plan()
    scan_path = OUT / "scan_pairs.csv"
    pairs = read_csv(scan_path)
    records = []
    image_cache = {}
    mask_cache = {}
    previous_crop = None
    for number, pair in enumerate(pairs, 1):
        if pair["mados_id"] != previous_crop:
            image_cache.clear()
            mask_cache.clear()
            previous_crop = pair["mados_id"]
        record = {**pair, **radiometry_and_labels(pair, image_cache, mask_cache)}
        records.append(record)
        if number % 100 == 0:
            print(f"Radiometry/labels {number}/{len(pairs)}", flush=True)
    if len(records) != len(pairs):
        raise RuntimeError("Pair analysis incomplete")
    columns = list(dict.fromkeys(key for record in records for key in record))
    write_csv(OUT / "dup_map_full.csv", records, columns)
    print("Wrote dup_map_full.csv", flush=True)
    regression_v1(records)


def regression_v1(records: list[dict] | None = None) -> None:
    if records is None:
        records = read_csv(OUT / "dup_map_full.csv")
    current = {(row["mados_id"], row["marida_patch"]): row for row in records}
    old_rad = read_csv(OUT.parent / "radiometry.csv")
    old_labels = read_csv(OUT.parent / "label_agreement.csv")
    if len(old_rad) != 342:
        raise RuntimeError("Expected 342 v1 radiometry pairs")
    differences = []
    for row in old_rad:
        key = row["mados_id"], row["marida_patch"]
        if key not in current:
            differences.append({"pair": key, "field": "missing", "v1": "present", "v2": "absent"})
            continue
        updated = current[key]
        expected_acq = "undetermined" if row["status"] == "insufficient_water" else row["same_acq"]
        if str(updated["same_acq"]) != expected_acq:
            differences.append({"pair": key, "field": "same_acq", "v1": expected_acq, "v2": updated["same_acq"]})
        if int(updated["n_water_px"]) != int(row["n_water_px"]):
            differences.append({"pair": key, "field": "n_water_px", "v1": row["n_water_px"], "v2": updated["n_water_px"]})
        if row["status"] == "ok":
            for band in t2.BANDS[:4]:
                for suffix in ("r", "slope"):
                    field = f"{band}_{suffix}"
                    a, b = float(row[field]), float(updated[field])
                    if not np.isclose(a, b, rtol=1e-7, atol=1e-8):
                        differences.append({"pair": key, "field": field, "v1": a, "v2": b})
    for row in old_labels:
        key = row["mados_id"], row["marida_patch"]
        updated = current.get(key)
        if updated is None:
            continue
        for field in ("px_mados", "px_marida", "px_intersection", "px_union",
                      "px_mados_only", "px_marida_only"):
            a, b = int(row[f"raw_{field}"]), int(updated[field])
            if a != b:
                differences.append({"pair": key, "field": field, "v1": a, "v2": b})
    report = {"v1_radiometry_pairs": len(old_rad), "v1_label_pairs": len(old_labels),
              "differences": differences[:100], "difference_count": len(differences),
              "tolerance": {"rtol": 1e-7, "atol": 1e-8}, "plan_sha256": locked_plan()}
    (OUT / "v1_pair_regression.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"V1 pair regression: {len(differences)} differences", flush=True)
    if differences:
        raise RuntimeError(f"V1 radiometry/label regression failed: {len(differences)} differences")


def negative_controls() -> None:
    locked_plan()
    cv2.setRNGSeed(0)
    cv2.setNumThreads(1)
    pairs = read_csv(OUT / "dup_map_full.csv")
    patches = marida_patches()
    index_by_stem = {path.stem: index for index, (path, _, _) in enumerate(patches)}
    tile_by_stem = {path.stem: scene.rsplit("_", 1)[-1] for path, scene, _ in patches}
    by_crop = defaultdict(list)
    for pair in pairs:
        by_crop[pair["mados_id"]].append(pair)
    split_order = official_splits()
    splits = {crop: split for split, ids in split_order.items() for crop in ids}
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    target_features = []
    for number, (path, _, _) in enumerate(patches, 1):
        image = matcher_code.load_marida(path)["B4"]
        kp, desc = sift.detectAndCompute(image, None)
        target_features.append((image.shape, kp, desc))
    rng = np.random.default_rng(NEGATIVE_SEED)
    rows = []
    for number, crop in enumerate(sorted(by_crop), 1):
        known_tiles = {tile_by_stem[pair["marida_patch"]] for pair in by_crop[crop]}
        candidates = [index for index, (path, scene, _) in enumerate(patches)
                      if scene.rsplit("_", 1)[-1] not in known_tiles]
        candidates.sort(key=lambda index: str(patches[index][0]))
        if len(candidates) < NEGATIVE_PER_CROP:
            raise RuntimeError(f"Insufficient negative controls for {crop}")
        images, _ = matcher_code.load_mados(crop)
        query = images["B4"]
        kp, desc = sift.detectAndCompute(query, None)
        query_features = (query.shape, kp, desc)
        source_bands = {}
        target_bands = {}
        selected = rng.choice(len(candidates), NEGATIVE_PER_CROP, replace=False)
        for choice in selected:
            result = score_pair(crop, candidates[int(choice)], query_features, target_features,
                                patches, source_bands, target_bands)
            rows.append({"mados_id": crop, "mados_split": splits[crop],
                         "marida_patch": result["marida_patch"], "marida_tile": result["marida_tile"],
                         "score": result["score"], "confirmed": result["confirmed"],
                         "geometry_pass": result["geometry_pass"], "integer_pass": result["integer_pass"],
                         "good_matches": result["good_matches"], "inliers": result["inliers"],
                         "b3_correlation": result["b3_correlation"],
                         "b2_correlation": result["b2_correlation"]})
        if number % 50 == 0:
            print(f"Negative-control crops {number}/{len(by_crop)}", flush=True)
    write_csv(OUT / "null_scores_full.csv", rows,
              ["mados_id", "mados_split", "marida_patch", "marida_tile", "score", "confirmed",
               "geometry_pass", "integer_pass", "good_matches", "inliers", "b3_correlation", "b2_correlation"])
    print(f"Wrote {len(rows)} negative-control scores", flush=True)


def coverage_union(pairs: list[dict], source: str) -> tuple[int, int, int, int]:
    windows = defaultdict(list)
    for pair in pairs:
        ar, ac, br, bc = t3.window(int(pair["dy"]), int(pair["dx"]))
        if source == "mados":
            windows[pair["mados_id"]].append((ar, ac))
        else:
            windows[pair["marida_patch"]].append((br, bc))
    positive_px = 0
    boxes = 0
    covered_px = 0
    total_px = 0
    for stem, slices in windows.items():
        mask = t3.load_mados_mask(stem) if source == "mados" else t3.load_marida_mask(stem)
        selected = np.zeros(mask.shape, dtype=bool)
        for rs, cs in slices:
            selected[rs, cs] = True
        positive_px += int((mask & selected).sum())
        covered_px += int(selected.sum())
        total_px += int(selected.size)
        labeled, eligible = t3.components(mask)
        for component in np.flatnonzero(eligible):
            coords = np.where(labeled == component)
            row = int(np.floor((int(coords[0].min()) + int(coords[0].max()) + 1) / 2))
            col = int(np.floor((int(coords[1].min()) + int(coords[1].max()) + 1) / 2))
            boxes += int(selected[row, col])
    return positive_px, boxes, covered_px, total_px


def pooled_agreement(pairs: list[dict]) -> dict:
    owned_a = {}
    owned_b = {}
    masks_a = {}
    masks_b = {}
    comps_a = {}
    comps_b = {}
    totals = Counter()
    seen_a = set()
    seen_b = set()
    hit_a = set()
    hit_b = set()
    for pair in sorted(pairs, key=lambda row: (row["mados_id"], -float(row["score"]), row["marida_patch"])):
        crop, patch = pair["mados_id"], pair["marida_patch"]
        if crop not in masks_a:
            masks_a[crop] = t3.load_mados_mask(crop)
            comps_a[crop] = t3.components(masks_a[crop])
            owned_a[crop] = np.zeros((240, 240), dtype=bool)
        if patch not in masks_b:
            masks_b[patch] = t3.load_marida_mask(patch)
            comps_b[patch] = t3.components(masks_b[patch])
            owned_b[patch] = np.zeros((256, 256), dtype=bool)
        ar, ac, br, bc = t3.window(int(pair["dy"]), int(pair["dx"]))
        selected = ~(owned_a[crop][ar, ac] | owned_b[patch][br, bc])
        owned_a[crop][ar, ac] |= selected
        owned_b[patch][br, bc] |= selected
        a = masks_a[crop][ar, ac]
        b = masks_b[patch][br, bc]
        totals.update(t3.tally(a, b, selected))
        la, ea = comps_a[crop]
        lb, eb = comps_b[patch]
        la, lb = la[ar, ac], lb[br, bc]
        for component in t3.component_ids(la, ea, selected):
            key = crop, int(component)
            seen_a.add(key)
            if np.any((la == component) & b & selected):
                hit_a.add(key)
        for component in t3.component_ids(lb, eb, selected):
            key = patch, int(component)
            seen_b.add(key)
            if np.any((lb == component) & a & selected):
                hit_b.add(key)
    return {**dict(totals),
            "pooled_px_iou": totals["px_intersection"] / totals["px_union"] if totals["px_union"] else None,
            "mados_components_touched": len(seen_a), "mados_components_hit_marida": len(hit_a),
            "marida_components_touched": len(seen_b), "marida_components_hit_mados": len(hit_b)}


def cross_split_ambiguity(pairs: list[dict]) -> dict:
    windows = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        ar, ac, _, _ = t3.window(int(pair["dy"]), int(pair["dx"]))
        windows[pair["mados_id"]][pair["marida_split"]].append((ar, ac))
    ambiguous_px = 0
    ambiguous_md_px = 0
    ambiguous_boxes = 0
    for crop, split_windows in windows.items():
        if len(split_windows) < 2:
            continue
        count = np.zeros((240, 240), dtype=np.uint8)
        for slices in split_windows.values():
            selected = np.zeros((240, 240), dtype=bool)
            for ar, ac in slices:
                selected[ar, ac] = True
            count += selected.astype(np.uint8)
        ambiguous = count >= 2
        mask = t3.load_mados_mask(crop)
        ambiguous_px += int(ambiguous.sum())
        ambiguous_md_px += int((ambiguous & mask).sum())
        labels, eligible = t3.components(mask)
        for component in np.flatnonzero(eligible):
            coords = np.where(labels == component)
            row = int(np.floor((int(coords[0].min()) + int(coords[0].max()) + 1) / 2))
            col = int(np.floor((int(coords[1].min()) + int(coords[1].max()) + 1) / 2))
            ambiguous_boxes += int(ambiguous[row, col])
    return {"mados_window_pixels_in_multiple_marida_splits": ambiguous_px,
            "mados_md_pixels_in_multiple_marida_splits": ambiguous_md_px,
            "mados_md_boxes_in_multiple_marida_splits": ambiguous_boxes}


def raw_input_manifest(pairs: list[dict]) -> dict:
    official = official_splits()
    confirmed_crops = {pair["mados_id"] for pair in pairs}
    confirmed_patches = {pair["marida_patch"] for pair in pairs}
    paths = set()
    tokens_by_scene = {}
    for ids in official.values():
        for crop_id in ids:
            scene, crop = t2.MADOS_ID.fullmatch(crop_id).groups()
            scene_dir = t2.MADOS_RAW / f"Scene_{scene}"
            if scene not in tokens_by_scene:
                tokens_by_scene[scene] = t2._discover_band_tokens(scene_dir)
            tokens = tokens_by_scene[scene]
            bands = t2.BANDS if crop_id in confirmed_crops else t2.BANDS[:3]
            for band in bands:
                folder = "10" if band in t2.BANDS[:4] else "20"
                paths.add(scene_dir / folder / f"Scene_{scene}_L2R_{tokens[band]}_{crop}.tif")
            if crop_id in confirmed_crops:
                paths.add(scene_dir / "10" / f"Scene_{scene}_L2R_cl_{crop}.tif")
    for patch_path, _, _ in marida_patches():
        paths.add(patch_path)
        if patch_path.stem in confirmed_patches:
            paths.add(patch_path.with_name(f"{patch_path.stem}_cl.tif"))
    manifest = OUT / "raw_input_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "mtime_ns", "sha256"])
        writer.writeheader()
        for number, path in enumerate(sorted(paths), 1):
            if not path.is_file():
                raise FileNotFoundError(path)
            stat = path.stat()
            writer.writerow({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                             "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                             "sha256": sha256(path)})
            if number % 1000 == 0:
                print(f"Hashed raw inputs {number}/{len(paths)}", flush=True)
    return {"files": len(paths), "manifest_sha256": sha256(manifest),
            "hash_method": "SHA-256 of every listed raw TIFF; manifest records path, size and mtime_ns"}


def finalize() -> None:
    plan_hash = locked_plan()
    regression = json.loads((OUT / "v1_pair_regression.json").read_text(encoding="utf-8"))
    if regression["difference_count"] != 0:
        raise RuntimeError("V1 pair regression must pass before finalizing")
    pairs = read_csv(OUT / "dup_map_full.csv")
    scan = read_csv(OUT / "scan_pairs.csv")
    if len(pairs) != len(scan) or len({(p["mados_id"], p["marida_patch"]) for p in pairs}) != len(pairs):
        raise RuntimeError("Pair map changed or contains duplicates")
    null = read_csv(OUT / "null_scores_full.csv")
    scan_progress = [json.loads(line) for line in (OUT / "scan_progress.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(scan_progress) != 2803:
        raise RuntimeError("Scan progress must contain all 2803 crops")
    retrieval_exclusions = {"no_descriptor_or_candidates": sum(
        not row["confirmed_pairs"] and row["candidates_checked"] == 0 for row in scan_progress),
        "top160_candidates_checked_without_confirmation": sum(
            not row["confirmed_pairs"] and row["candidates_checked"] > 0 for row in scan_progress)}
    by_cell = defaultdict(list)
    for pair in pairs:
        by_cell[pair["mados_split"], pair["marida_split"]].append(pair)
    rows = []
    for m_split in ("train", "val", "test"):
        for r_split in ("train", "val", "test"):
            cell = by_cell[m_split, r_split]
            confirmed = [pair for pair in cell if pair["same_acq"] == "True"]
            m_px, m_boxes, m_covered, m_total = coverage_union(confirmed, "mados")
            r_px, r_boxes, r_covered, r_total = coverage_union(confirmed, "marida")
            agreement = pooled_agreement(confirmed)
            rows.append({"mados_split": m_split, "marida_split": r_split,
                         "same_acq_pairs": len(confirmed),
                         "same_acq_mados_crops": len({pair["mados_id"] for pair in confirmed}),
                         "same_acq_marida_patches": len({pair["marida_patch"] for pair in confirmed}),
                         "mados_md_min4_boxes": m_boxes, "mados_md_pixels": m_px,
                         "marida_md_min4_boxes": r_boxes, "marida_md_pixels": r_px,
                         "mados_window_px": m_covered, "mados_total_patch_px": m_total,
                         "marida_window_px": r_covered, "marida_total_patch_px": r_total,
                         "pooled_px_intersection": agreement.get("px_intersection", 0),
                         "pooled_px_union": agreement.get("px_union", 0),
                         "pooled_px_iou": agreement["pooled_px_iou"],
                         "mados_components_touched": agreement["mados_components_touched"],
                         "mados_components_hit_marida": agreement["mados_components_hit_marida"],
                         "marida_components_touched": agreement["marida_components_touched"],
                         "marida_components_hit_mados": agreement["marida_components_hit_mados"],
                         "spatial_pairs_false": sum(pair["same_acq"] == "False" for pair in cell),
                         "spatial_pairs_undetermined": sum(pair["same_acq"] == "undetermined" for pair in cell),
                         "spatial_unique_mados_crops": len({pair["mados_id"] for pair in cell}),
                         "spatial_unique_marida_patches": len({pair["marida_patch"] for pair in cell})})
    write_csv(OUT / "dup_crosstab.csv", rows, list(rows[0]))
    splits = official_splits()
    per_split = {}
    for split in ("train", "val", "test"):
        scoped = [pair for pair in pairs if pair["mados_split"] == split]
        controls = [row for row in null if row["mados_split"] == split]
        scores = np.array([float(row["score"]) for row in controls])
        per_split[split] = {"denominator_crops": len(splits[split]), "spatial_pairs": len(scoped),
                            "spatial_unique_crops": len({pair["mados_id"] for pair in scoped}),
                            "same_acq_pairs": sum(pair["same_acq"] == "True" for pair in scoped),
                            "false_pairs": sum(pair["same_acq"] == "False" for pair in scoped),
                            "undetermined_pairs": sum(pair["same_acq"] == "undetermined" for pair in scoped),
                            "null_exceedances": int(np.sum(scores >= SCORE_THRESHOLD)),
                            "null_count": int(scores.size),
                            "null_threshold_empirical_percentile": float(np.mean(scores < SCORE_THRESHOLD)),
                            "null_quantiles": {str(q): float(np.quantile(scores, q)) for q in (0.5, 0.95, 0.99, 0.999, 1.0)}}
    all_null_scores = np.array([float(row["score"]) for row in null])
    overall_null = {"exceedances": int(np.sum(all_null_scores >= SCORE_THRESHOLD)),
                    "count": int(all_null_scores.size),
                    "threshold_empirical_percentile": float(np.mean(all_null_scores < SCORE_THRESHOLD)),
                    "quantiles": {str(q): float(np.quantile(all_null_scores, q)) for q in (0.5, 0.95, 0.99, 0.999, 1.0)}}
    same = [pair for pair in pairs if pair["same_acq"] == "True"]
    split_by_crop = defaultdict(set)
    for pair in same:
        split_by_crop[pair["mados_id"]].add(pair["marida_split"])
    ambiguous_rows = []
    for crop, matched_splits in sorted(split_by_crop.items()):
        if len(matched_splits) > 1:
            crop_pairs = [pair for pair in same if pair["mados_id"] == crop]
            ambiguous_rows.append({"mados_id": crop, "mados_split": crop_pairs[0]["mados_split"],
                                   "marida_splits": ";".join(sorted(matched_splits)),
                                   "marida_patches": ";".join(sorted({pair["marida_patch"] for pair in crop_pairs})),
                                   "same_acq_pairs": len(crop_pairs)})
    write_csv(OUT / "cross_split_crops.csv", ambiguous_rows,
              ["mados_id", "mados_split", "marida_splits", "marida_patches", "same_acq_pairs"])
    scene_by_crop = defaultdict(set)
    for pair in pairs:
        scene_by_crop[pair["mados_id"]].add(pair["marida_scene"])
    by_mados_scene = defaultdict(list)
    for pair in pairs:
        by_mados_scene[pair["mados_id"].rsplit("_", 1)[0]].append(pair)
    date_conflicts = []
    for mados_scene, scene_pairs in sorted(by_mados_scene.items()):
        dates = {pair["marida_scene"].rsplit("_", 1)[0] for pair in scene_pairs}
        if len(dates) > 1:
            date_conflicts.append({"mados_scene": mados_scene,
                                   "mados_crops": len({pair["mados_id"] for pair in scene_pairs}),
                                   "marida_dates": ";".join(sorted(dates)),
                                   "marida_scenes": ";".join(sorted({pair["marida_scene"] for pair in scene_pairs})),
                                   "spatial_pairs": len(scene_pairs),
                                   "same_acq_pairs": sum(pair["same_acq"] == "True" for pair in scene_pairs)})
    write_csv(OUT / "scene_date_conflicts.csv", date_conflicts,
              ["mados_scene", "mados_crops", "marida_dates", "marida_scenes", "spatial_pairs", "same_acq_pairs"])
    global_m_px, global_m_boxes, global_m_covered, global_m_total = coverage_union(same, "mados")
    global_r_px, global_r_boxes, global_r_covered, global_r_total = coverage_union(same, "marida")
    global_agreement = pooled_agreement(same)
    raw_manifest = raw_input_manifest(pairs)
    metadata = {"plan_sha256": plan_hash,
                "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
                    PLAN, OLD_MATCHES, matcher_code.MARIDA_INVENTORY,
                    *(t2.MADOS_RAW / "splits" / f"{split}_X.txt" for split in ("train", "val", "test")))},
                "script_sha256": sha256(Path(__file__)), "scan_pairs_sha256": sha256(OUT / "scan_pairs.csv"),
                "helper_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
                    Path(t1.__file__), Path(t2.__file__), Path(t3.__file__), Path(matcher_code.__file__),
                    ROOT / "scripts" / "mpb" / "materialize_mados_external.py")},
                "raw_input_manifest": raw_manifest,
                "scan_progress_sha256": sha256(OUT / "scan_progress.jsonl"),
                "scan_progress_created_utc": datetime.datetime.fromtimestamp(
                    (OUT / "scan_progress.jsonl").stat().st_ctime, datetime.timezone.utc).isoformat(),
                "dup_map_sha256": sha256(OUT / "dup_map_full.csv"),
                "v1_pair_regression_sha256": sha256(OUT / "v1_pair_regression.json"),
                "null_scores_sha256": sha256(OUT / "null_scores_full.csv"),
                "parameters": {"candidates": CANDIDATES, "sift_features": SIFT_FEATURES,
                               "flann_checks": 96, "score_threshold": SCORE_THRESHOLD,
                               "negative_seed": NEGATIVE_SEED, "negative_per_confirmed_crop": NEGATIVE_PER_CROP},
                "per_mados_split": per_split,
                "overall_null": overall_null,
                "retrieval_exclusions": retrieval_exclusions,
                "global": {"spatial_pairs": len(pairs),
                           "spatial_unique_mados_crops": len({pair["mados_id"] for pair in pairs}),
                           "same_acq_pairs": len(same),
                           "same_acq_unique_mados_crops": len({pair["mados_id"] for pair in same}),
                           "same_acq_unique_marida_patches": len({pair["marida_patch"] for pair in same}),
                           "cross_marida_split_same_acq_crops": sum(len(value) > 1 for value in split_by_crop.values()),
                           "multi_marida_scene_spatial_crops": sum(len(value) > 1 for value in scene_by_crop.values()),
                           "mados_scenes_with_multiple_marida_dates": len(date_conflicts),
                           "unique_mados_md_pixels": global_m_px, "unique_mados_md_boxes": global_m_boxes,
                           "unique_marida_md_pixels": global_r_px, "unique_marida_md_boxes": global_r_boxes,
                           "mados_window_coverage_px": global_m_covered,
                           "mados_source_total_px": global_m_total,
                           "marida_window_coverage_px": global_r_covered,
                           "marida_source_total_px": global_r_total,
                           "pooled_label_agreement": global_agreement,
                           "cross_split_ambiguity": cross_split_ambiguity(same)}}
    (OUT / "t6_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = [
        "# T6 全量重复图谱：方法与范围",
        "",
        f"锁定计划 SHA-256：`{plan_hash}`；脚本 SHA-256：`{metadata['script_sha256']}`。",
        "",
        "MADOS 官方 train/val/test 分母为 1433/642/728；MARIDA 以 MPB patch_inventory.csv 的 1381 个原始 patch 和 scene_split 为准。原始 TIFF 逐文件 SHA-256 清单见 raw_input_manifest.csv，帮助脚本哈希见 t6_summary.json。",
        "B4 各侧 p01/p99 映射 uint8，SIFT600、FLANN top160、Lowe 0.75、RANSAC 3px，几何及 B3/B2 相关阈值沿用 v1；match_score=min(B3,B2)，高置信阈值 0.95。整数偏移按 RANSAC 内点位移中位数 numpy.rint，旧 342 对全部回收才继续。",
        "同次获取且同处理仅是水体四个 10m 波段相关>0.98、MARIDA→MADOS 斜率偏离 1 不超过 5% 的操作性判据；水体不足 200 像元为未定，False 不直接证明不同日期。",
        "负对照在每个已确认裁块的确认瓦片号之外、按路径排序，以种子 20261015 无放回抽 20 个；同裁块分数相关，观测超阈率不能当全局误配概率。",
        "逐对原始 Marine Debris 使用双方 `_cl.tif` 类别 1，组件在完整原始 mask 上按 4 连通 MIN4 定义，再映入整数交叠窗口。",
        "交叉表主列仅计 same_acq=True；窗口在每个来源影像中求并集，像元 `(影像 ID,行,列)` 去重；框为完整组件外接框中心落在窗口并集内的原始 MIN4 组件。",
        "每格 pooled IoU 先将配对按 MADOS ID、score 降序、MARIDA patch 名排序，双方窗口像元占用一次，再汇总交并。这是“双侧唯一的配对像元”IoU，分母不能由各侧独立窗口并集像元数替代；组件触达率按原始 `(影像 ID,组件 ID)` 去重。",
        "各格可包含同一 MADOS 裁块在不同 MARIDA split 的重复覆盖，因此九格不能直接相加；全局唯一量和跨 split 重复覆盖见 t6_summary.json，裁块清单见 cross_split_crops.csv。不同 MARIDA 日期的空间配对另见 scene_date_conflicts.csv。",
        "未确认裁块分为无描述子或候选、以及 top160 已检查但未确认两类，计数见 t6_summary.json；扫描进度未保存逐候选几何/整数/B3B2 失败门槛，因此不能进一步拆解拒绝原因，也不能称未确认裁块独立。",
        "同瓦片跨日期的静态地物匹配需依赖辐射分层识别。",
        "",
    ]
    (OUT / "t6_method.md").write_text("\n".join(notes), encoding="utf-8")
    print(json.dumps(metadata["global"], ensure_ascii=False), flush=True)


def self_test() -> None:
    assert locked_plan() == "c548816f1bbffeef216f97a9a89cec02e55070267c476d3fa45134076b7a642d"
    assert {key: len(value) for key, value in official_splits().items()} == {"train": 1433, "val": 642, "test": 728}
    assert len(marida_patches()) == 1381
    assert len(old_pairs()) == 342
    assert t1.overlap_pixels((240, 240), (256, 256), 20, 22) == 55224
    assert pooled_agreement([])["pooled_px_iou"] is None
    assert cross_split_ambiguity([])["mados_md_boxes_in_multiple_marida_splits"] == 0
    synthetic = np.zeros((4, 4), dtype=bool)
    synthetic[1:3, 1:3] = True
    full_labels, full_eligible = t3.components(synthetic)
    assert t3.component_ids(full_labels[1:2, 1:3], full_eligible,
                            np.ones((1, 2), dtype=bool)).tolist() == [1]
    assert t3.component_ids(*t3.components(synthetic[1:2, 1:3]),
                            np.ones((1, 2), dtype=bool)).size == 0
    print("T6 self-test passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("self-test", "scan", "analyze", "regress", "null", "finalize", "all"), default="all")
    args = parser.parse_args()
    phases = ("scan", "analyze", "null", "finalize") if args.phase == "all" else (args.phase,)
    for phase in phases:
        globals()[{"self-test": "self_test", "scan": "scan", "analyze": "analyze", "regress": "regression_v1",
                   "null": "negative_controls", "finalize": "finalize"}[phase]]()


if __name__ == "__main__":
    main()
