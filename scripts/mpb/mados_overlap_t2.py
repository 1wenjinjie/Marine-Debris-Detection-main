"""Locked water-pixel radiometry and preprocessing audit for MADOS–MARIDA pairs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import rasterio
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mpb.materialize_mados_external import _discover_band_tokens, _load_crop_array
from scripts.mpb.run_experiments import _preprocess


OUT = ROOT / "runs_mpb" / "mados_overlap"
PLAN = OUT / "analysis_plan.md"
PLAN_HASH = OUT / "analysis_plan.sha256"
MATCHES = OUT / "matches.csv"
MARIDA_RAW = ROOT / "data" / "raw" / "MARIDA" / "patches"
MADOS_RAW = ROOT / "data" / "MADOS"
MARIDA_DATA = ROOT / "data" / "processed" / "marida_mpb_v1" / "datasets" / "scene_6ch_min4"
MADOS_DATA = ROOT / "data" / "processed" / "mados_external_min4" / "datasets" / "scene_6ch_min4"
BANDS = ("B4", "B3", "B2", "B8", "B11", "B12")
MARIDA_INDICES = (4, 3, 2, 8, 10, 11)
MADOS_ID = re.compile(r"^Scene_(\d+)_(\d+)$")
MIN_WATER = 200
MAX_RAW_MATERIALIZED_ERROR = 1e-4
MAX_INPUT_ERROR = 1 / 255 + 1e-4
MAX_RAW_NEAR_MEDIAN = 1e-4
MAX_RAW_NEAR_P95 = 1e-3
MIN_NORMALIZED_MISMATCH = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_plan_hash() -> str:
    actual = sha256(PLAN)
    expected = PLAN_HASH.read_text(encoding="utf-8").split()[0].lower()
    if actual != expected:
        raise RuntimeError(f"Analysis plan hash changed: expected {expected}, got {actual}")
    return actual


def load_mados_6b(mados_id: str) -> np.ndarray:
    match = MADOS_ID.fullmatch(mados_id)
    if match is None:
        raise ValueError(f"Invalid MADOS id: {mados_id}")
    scene, crop = match.groups()
    mados_scene = MADOS_RAW / f"Scene_{scene}"
    tokens = _discover_band_tokens(mados_scene)
    channels = []
    for band in BANDS:
        folder = "10" if band in BANDS[:4] else "20"
        path = mados_scene / folder / f"Scene_{scene}_L2R_{tokens[band]}_{crop}.tif"
        with rasterio.open(path) as source:
            array = source.read(1).astype(np.float32)
        if band not in BANDS[:4]:
            array = cv2.resize(array, (240, 240), interpolation=cv2.INTER_LINEAR)
        channels.append(array)
    stack = np.stack(channels)
    if stack.shape != (6, 240, 240):
        raise ValueError(f"Unexpected MADOS shape: {mados_id}: {stack.shape}")
    return stack


def load_marida_6b(marida_patch: str) -> np.ndarray:
    path = MARIDA_RAW / marida_patch.rsplit("_", 1)[0] / f"{marida_patch}.tif"
    with rasterio.open(path) as source:
        stack = source.read(MARIDA_INDICES).astype(np.float32)
    if stack.shape != (6, 256, 256):
        raise ValueError(f"Unexpected MARIDA shape: {marida_patch}: {stack.shape}")
    return stack


def image_path(dataset_root: Path, split: str, stem: str) -> Path:
    path = dataset_root / split / "images" / f"{stem}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def stats() -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    marida_path = MARIDA_DATA / "channel_stats_6ch.json"
    mados_path = MADOS_DATA / "channel_stats_6ch.json"
    hashes = {"marida_stats": sha256(marida_path), "mados_stats": sha256(mados_path)}
    if hashes["marida_stats"] != hashes["mados_stats"]:
        raise RuntimeError("MADOS and MARIDA model statistics differ")
    payload = json.loads(marida_path.read_text(encoding="utf-8"))
    if payload.get("fit_split") != "train_only" or payload.get("channels") != 6:
        raise RuntimeError("Expected locked six-channel train-only statistics")
    low = np.array(payload["percentiles"]["p01"], dtype=np.float32).reshape(1, 1, 6)
    high = np.array(payload["percentiles"]["p99"], dtype=np.float32).reshape(1, 1, 6)
    return low, high, hashes


def window(dy: int, dx: int, height: int = 240, width: int = 240) -> tuple[slice, slice, slice, slice] | None:
    r0, c0 = max(0, -dy), max(0, -dx)
    r1, c1 = min(height, 256 - dy), min(width, 256 - dx)
    if r1 <= r0 or c1 <= c0:
        return None
    return slice(r0, r1), slice(c0, c1), slice(r0 + dy, r1 + dy), slice(c0 + dx, c1 + dx)


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan
    mean_x, mean_y = x.mean(), y.mean()
    centered_x, centered_y = x - mean_x, y - mean_y
    denominator = float(np.dot(centered_x, centered_x))
    slope = float(np.dot(centered_x, centered_y) / denominator)
    intercept = float(mean_y - slope * mean_x)
    corr = float(np.dot(centered_x, centered_y) / np.sqrt(denominator * np.dot(centered_y, centered_y)))
    ratio = y[np.abs(x) > 1e-6] / x[np.abs(x) > 1e-6]
    return corr, slope, intercept, float(np.median(ratio)) if len(ratio) else np.nan


def normalized_native(array_hwc: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        (np.clip(array_hwc, low, high) - low) / np.maximum(high - low, np.float32(1e-6)),
        nan=0.0, posinf=1.0, neginf=0.0,
    ).astype(np.float32)


def expected_model_512(raw_hwc: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    native = normalized_native(raw_hwc, low, high)
    height, width = native.shape[:2]
    scale = 512 / max(height, width)
    resized_w, resized_h = round(width * scale), round(height * scale)
    resized = cv2.resize(native, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = 512 - resized_w, 512 - resized_h
    return np.pad(resized, ((pad_h // 2, pad_h - pad_h // 2),
                            (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                  constant_values=114 / 255).astype(np.float32)


def check_side(npy_path: Path, expected_materialized: np.ndarray,
               low: np.ndarray, high: np.ndarray) -> dict[str, float]:
    materialized = np.load(npy_path, allow_pickle=False).astype(np.float32)
    if materialized.shape != expected_materialized.shape:
        raise ValueError(f"Materialized shape changed: {npy_path}: {materialized.shape}")
    raw_error = float(np.nanmax(np.abs(materialized - expected_materialized)))
    actual_512, _ = _preprocess(npy_path, low, high, 512)
    expected_512 = expected_model_512(expected_materialized, low, high)
    input_error = float(np.nanmax(np.abs(actual_512 / 255 - expected_512)))
    return {"raw_materialized_max_abs": raw_error, "input_max_abs": input_error}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit() -> None:
    plan_hash = locked_plan_hash()
    if not MATCHES.is_file():
        raise FileNotFoundError(MATCHES)
    low, high, stats_hashes = stats()
    with MATCHES.open(newline="", encoding="utf-8-sig") as handle:
        matches = list(csv.DictReader(handle))
    radiometry = []
    preprocess = []
    counters = Counter()
    stop_signal = False
    for index, match in enumerate(matches, 1):
        if match.get("status") == "alignment_unresolved" or not match.get("dy") or not match.get("dx"):
            counters["alignment_unresolved"] += 1
            continue
        mados_id = match["mados_id"]
        marida_patch = match["marida_patch"]
        split = match["marida_split"]
        dy, dx = int(match["dy"]), int(match["dx"])
        slices = window(dy, dx)
        if slices is None:
            counters["empty_window"] += 1
            continue
        ar, ac, br, bc = slices
        a = load_mados_6b(mados_id)[:, ar, ac]
        b = load_marida_6b(marida_patch)[:, br, bc]
        denominator = b[1] + b[3]
        mask = np.isfinite(a).all(0) & np.isfinite(b).all(0) & (np.abs(denominator) > 1e-6)
        ndwi = np.divide(b[1] - b[3], denominator, out=np.full_like(denominator, np.nan),
                         where=np.abs(denominator) > 1e-6)
        mask &= ndwi > 0
        mask = ndimage.binary_erosion(mask, iterations=3, border_value=0)
        n = int(mask.sum())
        record = {"mados_id": mados_id, "marida_patch": marida_patch,
                  "marida_split": split, "split": split, "dy": dy, "dx": dx,
                  "n_overlap_px": int(a.shape[1] * a.shape[2]), "n_water_px": n,
                  "status": "ok" if n >= MIN_WATER else "insufficient_water",
                  "same_acq_same_processing": False, "same_acq": False}
        if n < MIN_WATER:
            counters["insufficient_water"] += 1
            radiometry.append(record)
            continue
        for i, band in enumerate(BANDS):
            corr, slope, intercept, medratio = fit(b[i][mask], a[i][mask])
            record.update({f"{band}_r": corr, f"{band}_slope": slope,
                           f"{band}_icpt": intercept, f"{band}_medratio": medratio})
        same = all(np.isfinite(record[f"{band}_r"]) and record[f"{band}_r"] > 0.98
                   and np.isfinite(record[f"{band}_slope"])
                   and abs(record[f"{band}_slope"] - 1) <= 0.05 for band in BANDS[:4])
        record["same_acq_same_processing"] = same
        record["same_acq"] = same
        counters["same_acq_same_processing" if same else "not_supported_same_acq_same_processing"] += 1
        radiometry.append(record)
        if not same:
            continue
        original_a = load_mados_6b(mados_id)
        original_b = load_marida_6b(marida_patch)
        mados_match = MADOS_ID.fullmatch(mados_id)
        assert mados_match is not None
        scene, crop = map(int, mados_match.groups())
        mados_expected = _load_crop_array(scene, crop, _discover_band_tokens(MADOS_RAW / f"Scene_{scene}"))
        marida_expected = original_b.transpose(1, 2, 0)
        a_path = image_path(MADOS_DATA, "test", mados_id)
        b_path = image_path(MARIDA_DATA, split, marida_patch)
        a_check = check_side(a_path, mados_expected, low, high)
        b_check = check_side(b_path, marida_expected, low, high)
        raw_difference = np.abs(original_a[:4, ar, ac] - original_b[:4, br, bc])[:, mask]
        raw_median = float(np.median(raw_difference))
        raw_p95 = float(np.percentile(raw_difference, 95))
        raw_near = raw_median <= MAX_RAW_NEAR_MEDIAN and raw_p95 <= MAX_RAW_NEAR_P95
        counters["raw_near"] += int(raw_near)
        a_native = normalized_native(np.load(a_path, allow_pickle=False), low, high).transpose(2, 0, 1)
        b_native = normalized_native(np.load(b_path, allow_pickle=False), low, high).transpose(2, 0, 1)
        normalized_difference = np.abs(a_native[:4, ar, ac] - b_native[:4, br, bc])[:, mask]
        normalized_median = float(np.median(normalized_difference))
        mismatch = raw_near and normalized_median > MIN_NORMALIZED_MISMATCH
        side_failure = any(check["raw_materialized_max_abs"] > MAX_RAW_MATERIALIZED_ERROR
                           or check["input_max_abs"] > MAX_INPUT_ERROR for check in (a_check, b_check))
        item = {"mados_id": mados_id, "marida_patch": marida_patch, "marida_split": split,
                "n_water_px": n, "raw_10m_abs_median": raw_median, "raw_10m_abs_p95": raw_p95,
                "raw_near": raw_near, "normalized_10m_abs_median": normalized_median,
                "preprocessing_mismatch": mismatch, "side_validation_failure": side_failure,
                "mados_raw_materialized_max_abs": a_check["raw_materialized_max_abs"],
                "marida_raw_materialized_max_abs": b_check["raw_materialized_max_abs"],
                "mados_input_max_abs": a_check["input_max_abs"],
                "marida_input_max_abs": b_check["input_max_abs"]}
        preprocess.append(item)
        counters["preprocessing_mismatch"] += int(mismatch)
        counters["side_validation_failure"] += int(side_failure)
        if mismatch or side_failure:
            stop_signal = True
            print(f"STOP: preprocessing inconsistency at {mados_id} / {marida_patch}", flush=True)
            break
        if index % 25 == 0:
            print(f"T2 {index}/{len(matches)} pairs", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    radiometry_fields = ["mados_id", "marida_patch", "marida_split", "split", "dy", "dx",
                         "n_overlap_px", "n_water_px", "status", "same_acq_same_processing", "same_acq"]
    for band in BANDS:
        radiometry_fields += [f"{band}_{suffix}" for suffix in ("r", "slope", "icpt", "medratio")]
    write_csv(OUT / "radiometry.csv", radiometry, radiometry_fields)
    preprocess_fields = ["mados_id", "marida_patch", "marida_split", "n_water_px", "raw_10m_abs_median",
                         "raw_10m_abs_p95", "raw_near", "normalized_10m_abs_median",
                         "preprocessing_mismatch", "side_validation_failure", "mados_raw_materialized_max_abs",
                         "marida_raw_materialized_max_abs", "mados_input_max_abs", "marida_input_max_abs"]
    write_csv(OUT / "preprocess_check.csv", preprocess, preprocess_fields)
    summary = {"analysis_plan_sha256": plan_hash, "matches_sha256": sha256(MATCHES),
               "script_sha256": sha256(Path(__file__)), "stats_sha256": stats_hashes,
               "matches_rows": len(matches), "radiometry_rows": len(radiometry),
               "preprocess_rows": len(preprocess), "counts": dict(counters),
               "stop_signal": stop_signal, "complete": not stop_signal and len(radiometry) + counters["alignment_unresolved"] + counters["empty_window"] == len(matches),
               "cross_dataset_preprocessing_test_eligible_pairs": counters["raw_near"]}
    (OUT / "t2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# T2 水体辐射与模型输入核查", "", f"- 计划 SHA-256：`{plan_hash}`",
             f"- 配对表 SHA-256：`{summary['matches_sha256']}`", f"- 处理配对：{len(radiometry)}/{len(matches)}",
             f"- 有效水体不足：{counters['insufficient_water']}",
             f"- 同次且同处理的操作性判定：{counters['same_acq_same_processing']}",
             f"- 不支持该判定：{counters['not_supported_same_acq_same_processing']}",
             f"- 逐侧校验失败：{counters['side_validation_failure']}",
             f"- 原始四个 10 m 波段达到预锁近一致门槛的配对：{counters['raw_near']}/{len(preprocess)}",
             f"- 原始近一致但模型输入不一致：{counters['preprocessing_mismatch']}",
             f"- 停止信号：{stop_signal}", "",
             "未通过阈值只表示当前数据不支持‘同次获取且同处理’，不能据此断定日期不同。",
             "B11/B12 仅作描述性比较；具体每对每波段数值见 radiometry.csv。",
             "若原始近一致合格配对数为 0，跨数据集预处理 bug 的条件判据未被触发；只能依据逐侧链路校验。"]
    (OUT / "radiometry_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if stop_signal:
        raise SystemExit(2)


def self_test() -> None:
    assert window(0, 0) == (slice(0, 240), slice(0, 240), slice(0, 240), slice(0, 240))
    assert window(-10, 20) == (slice(10, 240), slice(0, 236), slice(0, 230), slice(20, 256))
    x = np.arange(1, 101, dtype=np.float32)
    r, slope, intercept, ratio = fit(x, 2 * x + 3)
    assert abs(r - 1) < 1e-10 and abs(slope - 2) < 1e-10 and abs(intercept - 3) < 1e-10
    assert np.isfinite(ratio)
    print("T2 self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else audit()
