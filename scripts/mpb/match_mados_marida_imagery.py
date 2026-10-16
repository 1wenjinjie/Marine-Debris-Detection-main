"""Audit shared imagery between official MADOS test crops and raw MARIDA patches.

MADOS TIFFs have no trustworthy map transform. This audit therefore retrieves
candidate patches by B4 SIFT descriptors and validates them by RANSAC geometry
and independently sampled B3/B2 image agreement under the fitted transform.
Unconfirmed crops are inconclusive, never evidence of geographic independence.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[2]
MADOS_ROOT = ROOT / "data" / "MADOS"
MARIDA_ROOT = ROOT / "data" / "raw" / "MARIDA" / "patches"
MADOS_LABELS = ROOT / "data" / "processed" / "mados_external_min4" / "datasets" / "scene_6ch_min4" / "test" / "labels"
MARIDA_INVENTORY = ROOT / "data" / "processed" / "marida_mpb_v1" / "audit" / "patch_inventory.csv"
OUTPUT_ROOT = ROOT / "runs_mpb" / "overlap_audit"
MADOS_NAME = re.compile(r"^Scene_(\d+)_(\d+)$")
MARIDA_NAME = re.compile(r"^(S2_\d+-\d+-\d+_\w{5})_\d+$")
MIN_GOOD = 20
MIN_INLIERS = 12
MIN_INLIER_RATIO = 0.5
MAX_MEDIAN_ERROR_PX = 2.0
MIN_BAND_CORRELATION = 0.95
MIN_OVERLAP_PIXELS = 10000
MAX_TRANSLATION_DEVIATION_PX = 6.0


def normalized_band(array: np.ndarray) -> np.ndarray:
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return np.zeros(array.shape, np.uint8)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        return np.zeros(array.shape, np.uint8)
    return np.uint8(np.clip((np.nan_to_num(array, nan=lo) - lo) * 255.0 / (hi - lo), 0, 255))


def load_mados(crop_id: str, bands: tuple[str, ...] = ("B4",)) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    match = MADOS_NAME.fullmatch(crop_id)
    if not match:
        raise ValueError(f"Invalid MADOS crop id: {crop_id}")
    scene, index = match.groups()
    band_tokens = {"B4": "rhorc_665", "B3": "rhorc_559|rhorc_560", "B2": "rhorc_492"}
    source = {}
    images = {}
    for band in bands:
        choices = []
        for token in band_tokens[band].split("|"):
            choices.extend(sorted((MADOS_ROOT / f"Scene_{scene}" / "10").glob(f"*_{token}_{index}.tif")))
        if len(choices) != 1:
            raise FileNotFoundError(f"Expected one {band} image for {crop_id}, found {len(choices)}")
        source[band] = str(choices[0].resolve())
        with rasterio.open(choices[0]) as raster:
            images[band] = normalized_band(raster.read(1))
    return images, source


def load_marida(path: Path, bands: tuple[str, ...] = ("B4",)) -> dict[str, np.ndarray]:
    band_indices = {"B2": 2, "B3": 3, "B4": 4}
    with rasterio.open(path) as raster:
        return {band: normalized_band(raster.read(band_indices[band])) for band in bands}


def find_sources() -> tuple[list[str], list[str], list[tuple[Path, str, str]]]:
    official_test = [line.strip() for line in (MADOS_ROOT / "splits" / "test_X.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(official_test) != len(set(official_test)):
        raise RuntimeError("Duplicate crop ids in official MADOS test split")
    positives = []
    for crop_id in official_test:
        label_path = MADOS_LABELS / f"{crop_id}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        if any(line.split() and line.split()[0] == "0" for line in label_path.read_text(encoding="utf-8").splitlines()):
            positives.append(crop_id)
    inventory = {}
    with MARIDA_INVENTORY.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            inventory[row["stem"]] = (row["scene"], row["scene_split"])
    paths = sorted(MARIDA_ROOT.glob("*/*.tif"))
    patches = []
    for path in paths:
        if path.stem not in inventory:
            continue
        scene, split = inventory[path.stem]
        patches.append((path, scene, split))
    if len(patches) != len(inventory):
        raise RuntimeError(f"Only {len(patches)} of {len(inventory)} MARIDA inventory patches found")
    return official_test, positives, patches


def retrieve_candidates(query: np.ndarray, matcher: cv2.FlannBasedMatcher, limit: int) -> list[tuple[int, int]]:
    hits = Counter()
    if len(query) < 2:
        return []
    for neighbours in matcher.knnMatch(query, k=4):
        seen = set()
        for hit in neighbours:
            if hit.imgIdx not in seen:
                hits[hit.imgIdx] += 1
                seen.add(hit.imgIdx)
    return hits.most_common(limit)


def band_correlation(source: np.ndarray, target: np.ndarray, homography: np.ndarray) -> tuple[float, int]:
    height, width = target.shape
    warped = cv2.warpPerspective(source, homography, (width, height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(source.shape, 255, np.uint8), homography, (width, height), flags=cv2.INTER_NEAREST) > 0
    n_pixels = int(mask.sum())
    if n_pixels < MIN_OVERLAP_PIXELS:
        return float("nan"), n_pixels
    first = cv2.GaussianBlur(warped, (5, 5), 1.0)[mask].astype(np.float32)
    second = cv2.GaussianBlur(target, (5, 5), 1.0)[mask].astype(np.float32)
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-6:
        return float("nan"), n_pixels
    return float(np.dot(first, second) / denominator), n_pixels


def translation_deviation(homography: np.ndarray, shape: tuple[int, int]) -> float:
    height, width = shape
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    shifts = projected - corners.reshape(-1, 2)
    center = np.median(shifts, axis=0)
    return float(np.max(np.linalg.norm(shifts - center, axis=1)))


def validate_pair(query_image: np.ndarray, query_kp: list, query_desc: np.ndarray,
                  target_image: np.ndarray, target_kp: list, target_desc: np.ndarray,
                  ratio: float, ransac_px: float) -> dict:
    result = {"good_matches": 0, "inliers": 0, "inlier_ratio": 0.0,
              "median_reprojection_error_px": None, "homography": None}
    if query_desc is None or target_desc is None or len(query_desc) < 2 or len(target_desc) < 2:
        return result
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    good = [first for first, second in matcher.knnMatch(query_desc, target_desc, k=2) if first.distance < ratio * second.distance]
    result["good_matches"] = len(good)
    if len(good) < 4:
        return result
    source_pts = np.float32([query_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    target_pts = np.float32([target_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source_pts, target_pts, cv2.RANSAC, ransac_px)
    if homography is None or mask is None:
        return result
    inlier = mask.ravel().astype(bool)
    result["inliers"] = int(inlier.sum())
    result["inlier_ratio"] = round(float(inlier.mean()), 6)
    projected = cv2.perspectiveTransform(source_pts, homography)
    errors = np.linalg.norm(projected - target_pts, axis=2).ravel()
    result["median_reprojection_error_px"] = round(float(np.median(errors[inlier])), 6) if inlier.any() else None
    result["homography"] = homography
    return result


def summarize_results(rows: list[dict], confirmed_pairs: list[dict]) -> dict:
    scene_map = {}
    crop_splits = {}
    for pair in confirmed_pairs:
        scene_map.setdefault(pair["mados_scene"], set()).add(pair["marida_scene"])
        crop_splits.setdefault(pair["marida_split"], set()).add(pair["mados_crop"])
    conflicts = {scene: sorted(matches) for scene, matches in scene_map.items() if len(matches) > 1}
    for row in rows:
        row["scene_mapping_conflict"] = row["mados_scene"] in conflicts
    return {"confirmed_crops": [row["mados_crop"] for row in rows if row["status"] == "confirmed_same_imagery"],
            "confirmed_pair_count_by_marida_split": dict(Counter(pair["marida_split"] for pair in confirmed_pairs)),
            "confirmed_unique_crop_count_by_marida_split": {split: len(crops) for split, crops in sorted(crop_splits.items())},
            "preferred_confirmed_crop_count_by_marida_split": dict(Counter(row["marida_split"] for row in rows if row["status"] == "confirmed_same_imagery")),
            "inconclusive_crops": [row["mados_crop"] for row in rows if row["status"] != "confirmed_same_imagery"],
            "mados_scene_to_marida_scene_conflicts": conflicts}


def audit(args: argparse.Namespace) -> None:
    cv2.setRNGSeed(0)
    cv2.setNumThreads(1)
    official_test, positives, patches = find_sources()
    audit_set = positives if args.debris_positive_only else official_test
    chosen = audit_set if args.max_crops is None else audit_set[:args.max_crops]
    sift = cv2.SIFT_create(nfeatures=args.sift_features)
    marida_features = []
    descriptors = []
    for index, (path, _, _) in enumerate(patches):
        image = load_marida(path)["B4"]
        keypoints, descriptor = sift.detectAndCompute(image, None)
        marida_features.append((keypoints, descriptor))
        descriptors.append(descriptor if descriptor is not None and len(descriptor) >= 2 else np.zeros((1, 128), np.float32))
        if (index + 1) % 200 == 0:
            print(f"Indexed MARIDA patches: {index + 1}/{len(patches)}", flush=True)
    index = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=args.flann_checks))
    index.add(descriptors)
    index.train()
    rows = []
    confirmed_pairs = []
    for number, crop_id in enumerate(chosen, 1):
        query_b4, source_paths = load_mados(crop_id)
        query_keypoints, query_descriptor = sift.detectAndCompute(query_b4["B4"], None)
        candidates = retrieve_candidates(query_descriptor, index, args.candidates) if query_descriptor is not None else []
        ranked = []
        for candidate_idx, hits in candidates:
            patch, scene, split = patches[candidate_idx]
            target_b4 = load_marida(patch)["B4"]
            keypoints, descriptor = marida_features[candidate_idx]
            check = validate_pair(query_b4["B4"], query_keypoints, query_descriptor,
                                  target_b4, keypoints, descriptor, args.ratio, args.ransac_px)
            if check["good_matches"] < MIN_GOOD or check["inliers"] < MIN_INLIERS:
                continue
            check.update({"candidate_idx": candidate_idx, "retrieval_hits": hits})
            ranked.append(check)
        ranked.sort(key=lambda value: (value["inliers"], value["good_matches"]), reverse=True)
        row = {"mados_crop": crop_id, "mados_scene": crop_id.rsplit("_", 1)[0], "mados_split": "test",
               "mados_b4_path": source_paths["B4"], "marida_patch": "", "marida_scene": "",
               "marida_split": "", "marida_path": "", "retrieval_hits": 0, "good_matches": 0,
               "inliers": 0, "inlier_ratio": 0.0, "median_reprojection_error_px": "",
               "b3_correlation": "", "b2_correlation": "", "overlap_pixels": 0,
               "translation_deviation_px": "", "mados_debris_positive_min4": crop_id in positives,
               "status": "inconclusive", "candidates_checked": len(candidates),
               "confirmed_candidate_count": 0, "confirmed_marida_patches": "", "confirmed_marida_splits": ""}
        best_attempt = None
        confirmed = []
        query_bands = None
        for check in ranked:
            patch, scene, split = patches[check["candidate_idx"]]
            deviation = translation_deviation(check["homography"], query_b4["B4"].shape)
            candidate = {"marida_patch": patch.stem, "marida_scene": scene, "marida_split": split,
                         "marida_path": str(patch.resolve()), "retrieval_hits": check["retrieval_hits"],
                         "good_matches": check["good_matches"], "inliers": check["inliers"],
                         "inlier_ratio": check["inlier_ratio"],
                         "median_reprojection_error_px": check["median_reprojection_error_px"],
                         "translation_deviation_px": round(deviation, 6), "b3_correlation": "",
                         "b2_correlation": "", "overlap_pixels": 0}
            if (check["inlier_ratio"] >= MIN_INLIER_RATIO
                    and check["median_reprojection_error_px"] <= MAX_MEDIAN_ERROR_PX
                    and deviation <= MAX_TRANSLATION_DEVIATION_PX):
                if query_bands is None:
                    query_bands, _ = load_mados(crop_id, ("B3", "B2"))
                target_bands = load_marida(patch, ("B3", "B2"))
                for band in ("B3", "B2"):
                    correlation, overlap = band_correlation(query_bands[band], target_bands[band], check["homography"])
                    candidate[f"{band.lower()}_correlation"] = round(correlation, 6) if np.isfinite(correlation) else ""
                    candidate["overlap_pixels"] = overlap
                if candidate["overlap_pixels"] >= MIN_OVERLAP_PIXELS and all(isinstance(candidate[f"{band}_correlation"], float) and candidate[f"{band}_correlation"] >= MIN_BAND_CORRELATION
                       for band in ("b3", "b2")):
                    confirmed.append(candidate)
                    confirmed_pairs.append({"mados_crop": crop_id, "mados_scene": row["mados_scene"], "mados_split": "test", **candidate})
            if best_attempt is None:
                best_attempt = candidate
        selected = confirmed[0] if confirmed else best_attempt
        if selected:
            row.update(selected)
        if confirmed:
            row["status"] = "confirmed_same_imagery"
            row["confirmed_candidate_count"] = len(confirmed)
            row["confirmed_marida_patches"] = ";".join(item["marida_patch"] for item in confirmed)
            row["confirmed_marida_splits"] = ";".join(sorted({item["marida_split"] for item in confirmed}))
        rows.append(row)
        print(f"{number}/{len(chosen)} {crop_id}: {row['status']} {row['marida_patch']} "
              f"inliers={row['inliers']} B3={row['b3_correlation']} B2={row['b2_correlation']}", flush=True)
    summary = summarize_results(rows, confirmed_pairs)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "image_matches.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    payload = {"method": "B4 SIFT global descriptor retrieval; local Lowe-ratio matches; RANSAC homography; warped B3/B2 Pearson correlation",
               "inputs": {"scan_scope": "debris_positive_min4_only" if args.debris_positive_only else "all_official_test_crops",
                          "mados_official_test_crops": len(official_test),
                          "mados_test_debris_positive_min4": len(positives), "mados_crops_audited": len(chosen),
                          "marida_raw_patches_indexed": len(patches), "marida_splits": dict(Counter(split for _, _, split in patches))},
               "parameters": {"sift_features": args.sift_features, "flann_checks": args.flann_checks,
                              "candidate_limit_per_crop": args.candidates, "lowe_ratio": args.ratio,
                              "ransac_px": args.ransac_px},
               "confirmation_thresholds": {"min_good_matches": MIN_GOOD, "min_inliers": MIN_INLIERS,
                                           "min_inlier_ratio": MIN_INLIER_RATIO, "max_median_reprojection_error_px": MAX_MEDIAN_ERROR_PX,
                                           "min_b3_and_b2_correlation": MIN_BAND_CORRELATION,
                                           "min_overlap_pixels": MIN_OVERLAP_PIXELS,
                                           "max_translation_deviation_px": MAX_TRANSLATION_DEVIATION_PX},
               "results": summary,
               "not_audited": {"mados_test_without_debris_min4": len(official_test) - len(positives),
                               "mados_test_crops_not_audited": [crop for crop in official_test if crop not in set(chosen)],
                               "candidate_search": f"Top {args.candidates} retrieved MARIDA patches per audited MADOS crop; failure to confirm is not proof of independence."},
               "rows": rows, "confirmed_pairs": confirmed_pairs}
    (args.output / "image_matches.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {args.output / 'image_matches.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--all-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--debris-positive-only", action="store_true", help="Limit audit to official test crops with MIN4 debris labels")
    parser.add_argument("--max-crops", type=int, default=None)
    parser.add_argument("--sift-features", type=int, default=600)
    parser.add_argument("--flann-checks", type=int, default=96)
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-px", type=float, default=3.0)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
