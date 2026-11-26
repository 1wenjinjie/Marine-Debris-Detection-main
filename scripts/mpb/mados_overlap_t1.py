"""Export integer MADOS–MARIDA alignments and cross-tile negative controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from match_mados_marida_imagery import (
    MAX_MEDIAN_ERROR_PX,
    MAX_TRANSLATION_DEVIATION_PX,
    MIN_BAND_CORRELATION,
    MIN_GOOD,
    MIN_INLIER_RATIO,
    MIN_INLIERS,
    MIN_OVERLAP_PIXELS,
    ROOT,
    band_correlation,
    find_sources,
    load_mados,
    load_marida,
    translation_deviation,
)


OUTPUT = ROOT / "runs_mpb" / "mados_overlap"
PRIOR = ROOT / "runs_mpb" / "overlap_audit" / "all_test" / "image_matches.json"
PLAN = OUTPUT / "analysis_plan.md"
SIFT_FEATURES = 600
LOWE_RATIO = 0.75
RANSAC_PX = 3.0
INTEGER_RESIDUAL_PX = 1.5
HIGH_CONFIDENCE = 0.95
NEGATIVE_PER_CROP = 20
NEGATIVE_SEED = 20261014
MATCH_METHOD = "B4_SIFT600_Lowe075_RANSAC3_B3B2corr_integer_translation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlap_pixels(shape_a: tuple[int, int], shape_b: tuple[int, int], dy: int, dx: int) -> int:
    h_a, w_a = shape_a
    h_b, w_b = shape_b
    height = min(h_a, h_b - dy) - max(0, -dy)
    width = min(w_a, w_b - dx) - max(0, -dx)
    return max(height, 0) * max(width, 0)


def fit_pair(query_kp: list, query_desc: np.ndarray | None,
             target_kp: list, target_desc: np.ndarray | None,
             query_shape: tuple[int, int], target_shape: tuple[int, int]) -> dict:
    result = {"good_matches": 0, "inliers": 0, "inlier_ratio": 0.0,
              "median_reprojection_error_px": np.nan, "translation_deviation_px": np.nan,
              "integer_residual_px": np.nan, "dy": None, "dx": None,
              "n_overlap_px": 0, "homography": None, "geometry_pass": False,
              "integer_pass": False}
    if query_desc is None or target_desc is None or len(query_desc) < 2 or len(target_desc) < 2:
        return result
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query_desc, target_desc, k=2)
    good = [first for first, second in matches if first.distance < LOWE_RATIO * second.distance]
    result["good_matches"] = len(good)
    if len(good) < 4:
        return result
    source = np.float32([query_kp[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    target = np.float32([target_kp[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, RANSAC_PX)
    if homography is None or mask is None:
        return result
    inlier = mask.ravel().astype(bool)
    result["inliers"] = int(inlier.sum())
    result["inlier_ratio"] = float(inlier.mean())
    projected = cv2.perspectiveTransform(source, homography)
    error = np.linalg.norm(projected - target, axis=2).ravel()
    if inlier.any():
        result["median_reprojection_error_px"] = float(np.median(error[inlier]))
    result["translation_deviation_px"] = translation_deviation(homography, query_shape)
    result["homography"] = homography
    result["geometry_pass"] = bool(
        result["good_matches"] >= MIN_GOOD
        and result["inliers"] >= MIN_INLIERS
        and result["inlier_ratio"] >= MIN_INLIER_RATIO
        and result["median_reprojection_error_px"] <= MAX_MEDIAN_ERROR_PX
        and result["translation_deviation_px"] <= MAX_TRANSLATION_DEVIATION_PX
    )
    if not result["geometry_pass"]:
        return result
    displacement = (target - source).reshape(-1, 2)[inlier]
    dx = int(np.rint(np.median(displacement[:, 0])))
    dy = int(np.rint(np.median(displacement[:, 1])))
    residual = np.linalg.norm(displacement - np.array([dx, dy]), axis=1)
    result["integer_residual_px"] = float(np.median(residual))
    result["dx"] = dx
    result["dy"] = dy
    result["n_overlap_px"] = overlap_pixels(query_shape, target_shape, dy, dx)
    result["integer_pass"] = bool(
        result["integer_residual_px"] <= INTEGER_RESIDUAL_PX
        and result["n_overlap_px"] >= MIN_OVERLAP_PIXELS
    )
    return result


def safe_number(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6) if np.isfinite(value) else ""
    return "" if value is None else value


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit(args: argparse.Namespace) -> None:
    cv2.setRNGSeed(0)
    cv2.setNumThreads(1)
    prior = json.loads(args.prior.read_text(encoding="utf-8"))
    pairs = prior["confirmed_pairs"]
    if len(pairs) != 342 or len({item["mados_crop"] for item in pairs}) != 285:
        raise RuntimeError("Expected the locked 342 confirmed pairs from 285 MADOS crops")
    if prior["parameters"]["sift_features"] != SIFT_FEATURES or prior["parameters"]["lowe_ratio"] != LOWE_RATIO:
        raise RuntimeError("Prior matcher parameters differ from analysis plan")
    _, _, patches = find_sources()
    paths = {path.stem: path for path, _, _ in patches}
    split = {path.stem: scene_split for path, _, scene_split in patches}
    scene = {path.stem: scene_name for path, scene_name, _ in patches}
    confirmed_tiles = defaultdict(set)
    for pair in pairs:
        confirmed_tiles[pair["mados_crop"]].add(pair["marida_scene"].rsplit("_", 1)[-1])
    conflicts = prior["results"].get("mados_scene_to_marida_scene_conflicts", {})
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    target_features = {}
    source_features = {}
    source_bands = {}
    target_bands = {}

    def get_source(crop_id: str):
        if crop_id not in source_features:
            images, _ = load_mados(crop_id)
            source_features[crop_id] = (images["B4"].shape, *sift.detectAndCompute(images["B4"], None))
        return source_features[crop_id]

    def get_target(stem: str):
        if stem not in target_features:
            image = load_marida(paths[stem])["B4"]
            target_features[stem] = (image.shape, *sift.detectAndCompute(image, None))
        return target_features[stem]

    def score(crop_id: str, stem: str) -> dict:
        query_shape, query_kp, query_desc = get_source(crop_id)
        target_shape, target_kp, target_desc = get_target(stem)
        fit = fit_pair(query_kp, query_desc, target_kp, target_desc, query_shape, target_shape)
        b3_corr = np.nan
        b2_corr = np.nan
        warped_overlap = 0
        match_score = 0.0
        if fit["geometry_pass"]:
            if crop_id not in source_bands:
                source_bands[crop_id], _ = load_mados(crop_id, ("B3", "B2"))
            if stem not in target_bands:
                target_bands[stem] = load_marida(paths[stem], ("B3", "B2"))
            b3_corr, warped_overlap = band_correlation(source_bands[crop_id]["B3"], target_bands[stem]["B3"], fit["homography"])
            b2_corr, _ = band_correlation(source_bands[crop_id]["B2"], target_bands[stem]["B2"], fit["homography"])
            if warped_overlap >= MIN_OVERLAP_PIXELS and np.isfinite(b3_corr) and np.isfinite(b2_corr):
                match_score = min(b3_corr, b2_corr)
        result = {key: safe_number(value) for key, value in fit.items() if key != "homography"}
        result.update({"b3_correlation": safe_number(b3_corr), "b2_correlation": safe_number(b2_corr),
                       "warped_overlap_px": warped_overlap, "match_score": round(match_score, 6),
                       "high_confidence": match_score >= HIGH_CONFIDENCE})
        return result

    match_rows = []
    for number, pair in enumerate(pairs, 1):
        crop_id = pair["mados_crop"]
        stem = pair["marida_patch"]
        result = score(crop_id, stem)
        status = "valid" if result["high_confidence"] and result["integer_pass"] else (
            "alignment_unresolved" if result["high_confidence"] else "reconfirmation_failed"
        )
        match_rows.append({"mados_id": crop_id, "marida_patch": stem, "marida_split": split[stem],
                           "dy": result["dy"] if status == "valid" else "",
                           "dx": result["dx"] if status == "valid" else "",
                           "match_score": result["match_score"], "match_method": MATCH_METHOD,
                           "n_overlap_px": result["n_overlap_px"] if status == "valid" else 0,
                           "status": status, "mados_scene": pair["mados_scene"],
                           "marida_scene": scene[stem], "marida_tile": scene[stem].rsplit("_", 1)[-1],
                           "scene_mapping_conflict": pair["mados_scene"] in conflicts,
                           **{key: result[key] for key in ("good_matches", "inliers", "inlier_ratio",
                                "median_reprojection_error_px", "translation_deviation_px",
                                "integer_residual_px", "b3_correlation", "b2_correlation",
                                "warped_overlap_px")}})
        if number % 50 == 0:
            print(f"Confirmed pairs refitted: {number}/{len(pairs)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    match_columns = ["mados_id", "marida_patch", "marida_split", "dy", "dx", "match_score",
                     "match_method", "n_overlap_px", "status", "mados_scene", "marida_scene",
                     "marida_tile", "scene_mapping_conflict", "good_matches", "inliers", "inlier_ratio",
                     "median_reprojection_error_px", "translation_deviation_px", "integer_residual_px",
                     "b3_correlation", "b2_correlation", "warped_overlap_px"]
    write_csv(args.output / "matches.csv", match_rows, match_columns)
    print(f"Wrote matches.csv: {dict(Counter(row['status'] for row in match_rows))}", flush=True)

    random = np.random.default_rng(NEGATIVE_SEED)
    ordered_patches = sorted(patches, key=lambda item: str(item[0]))
    null_rows = []
    for number, crop_id in enumerate(sorted(confirmed_tiles), 1):
        candidates = [path.stem for path, scene_name, _ in ordered_patches
                      if scene_name.rsplit("_", 1)[-1] not in confirmed_tiles[crop_id]]
        if len(candidates) < NEGATIVE_PER_CROP:
            raise RuntimeError(f"Not enough different-tile controls for {crop_id}")
        selected = random.choice(len(candidates), size=NEGATIVE_PER_CROP, replace=False)
        for index in selected:
            stem = candidates[int(index)]
            result = score(crop_id, stem)
            null_rows.append({"mados_id": crop_id, "marida_patch": stem,
                              "marida_tile": scene[stem].rsplit("_", 1)[-1],
                              "match_score": result["match_score"],
                              "high_confidence": result["high_confidence"],
                              "geometry_pass": result["geometry_pass"],
                              "integer_pass": result["integer_pass"],
                              "good_matches": result["good_matches"], "inliers": result["inliers"],
                              "b3_correlation": result["b3_correlation"],
                              "b2_correlation": result["b2_correlation"]})
        if number % 25 == 0:
            print(f"Negative-control crops: {number}/{len(confirmed_tiles)}", flush=True)

    null_columns = ["mados_id", "marida_patch", "marida_tile", "match_score", "high_confidence",
                    "geometry_pass", "integer_pass", "good_matches", "inliers", "b3_correlation",
                    "b2_correlation"]
    write_csv(args.output / "null_scores.csv", null_rows, null_columns)
    null_scores = np.asarray([row["match_score"] for row in null_rows], dtype=float)
    false_hits = int(np.count_nonzero(null_scores >= HIGH_CONFIDENCE))
    n_null = len(null_rows)
    upper = 1 - 0.05 ** (1 / n_null) if false_hits == 0 else None
    quality = Counter(row["status"] for row in match_rows)
    quality_splits = Counter(row["marida_split"] for row in match_rows if row["status"] == "valid")
    notes = [
        "# T1 匹配方法与负对照",
        "",
        f"- 预注册计划 SHA-256：`{sha256(PLAN)}`。",
        f"- 已有匹配结果 SHA-256：`{sha256(args.prior)}`。",
        f"- 本脚本 SHA-256：`{sha256(Path(__file__))}`。",
        f"- 输入：已有 {len(pairs)} 对确认配对、{len(confirmed_tiles)} 个 MADOS 测试裁块；MARIDA inventory {len(patches)} 个原始 patch。",
        "- 匹配：两边 B4 各自 p01/p99 映射到 uint8；SIFT 最多 600 特征，BF L2 最近邻 Lowe 比率 0.75，RANSAC 3 px。",
        "- 门槛：good≥20、inliers≥12、inlier ratio≥0.5、中位重投影误差≤2 px、四角平移偏差≤6 px、透视重叠≥10000 像元；B3/B2 分别独立归一化并以 5×5 GaussianBlur 后在单应性下相关，均≥0.95。",
        "- H 方向：MADOS→MARIDA。`dy,dx` 为 RANSAC 内点目标减来源坐标的各轴中位数经 `numpy.rint` 取整；整数位移残差中位数≤1.5 px，整数窗口重叠≥10000 像元。未通过者不进入 T2–T4。",
        "- `match_score`：几何门槛通过且透视重叠≥10000 时取 B3/B2 相关的较小值；否则 0。高置信阈值固定为 0.95。",
        f"- 重新拟合结果：{dict(quality)}；有效配对按 MARIDA split：{dict(quality_splits)}。",
        f"- 负对照：按路径排序、随机种子 {NEGATIVE_SEED}；每个已匹配 MADOS 裁块从所有确认瓦片以外的 MARIDA patch 中无放回抽取 {NEGATIVE_PER_CROP} 个，并跑相同评分过程。共 {n_null} 对。",
        f"- 零分布阈值 ≥0.95：{false_hits}/{n_null}（经验比例 {false_hits/n_null:.8f}）；阈值经验分位 `P(score<0.95)` = {np.mean(null_scores < HIGH_CONFIDENCE):.8f}。",
        f"- 零分布分位数：{dict(zip(['p50', 'p95', 'p99', 'p99.9', 'max'], [round(float(value), 6) for value in np.quantile(null_scores, [0.5, 0.95, 0.99, 0.999, 1.0])]))}。",
    ]
    if upper is not None:
        notes.append(f"- 0 次超阈时单侧 95% 二项上界：`1-0.05^(1/{n_null})` = {upper:.8f}；不能称真实假阳性率为 0。")
    notes.extend(["- 异瓦片负对照仅估计此采样方案下的随机误匹配；不能发现同一 MGRS 瓦片内的不同日期静态地物匹配，也不能证明获取日期。",
                  "- `matches.csv` 保留全部原确认对，包括配准未解和重新确认失败记录；T2–T4 只使用 `status=valid`。",
                  ""])
    (args.output / "matching_method.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"Matches: {dict(quality)}", flush=True)
    print(f"Null exceedances: {false_hits}/{n_null}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=PRIOR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
