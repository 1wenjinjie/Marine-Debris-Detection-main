"""Extract the real-data elements of MPB Fig. 1 (read-only on data).

Writes only runs_v2/_tmp_eval/fig1/{fig1_extract.txt, fig1_selection.json}.
Selection rules are fixed in code before any image is viewed:
  (a) scene = MARIDA scene with the most MIN4 debris instances in
      cv9_fold_assignment_actual.csv (ties: more patches, then scene name).
      All patches of that scene: true colour B4/B3/B2, one linear 2-98 %
      stretch shared by all its patches, 16 x 16 block-mean thumbnails,
      UTM bounds of each patch.
  (c) pair = dup_map_full.csv rows with same_acq True, mados_split test,
      marida_split train; largest px_intersection (ties: score desc,
      mados_id, marida_patch). 64 x 64 px window centred on the MARIDA
      class-1 pixels inside the overlap; MADOS window = same ground window
      through the integer offset (dx, dy). The offset sign is chosen by the
      larger B4 correlation and logged. One 2-98 % stretch shared by both.
  map = per-scene centroid (mean of patch centres), n_patches, CV fold.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform

ROOT = Path(__file__).resolve().parents[1]
MARIDA = ROOT / "data" / "raw" / "MARIDA" / "patches"
MADOS = ROOT / "data" / "MADOS"
FOLDS = ROOT / "runs_v2" / "_tmp_eval" / "fdi_baseline" / "cv9_fold_assignment_actual.csv"
DUP = ROOT / "runs_mpb" / "mados_overlap" / "v2" / "dup_map_full.csv"
INV = ROOT / "data" / "processed" / "marida_mpb_v1" / "audit" / "patch_inventory.csv"
OUT = ROOT / "runs_v2" / "_tmp_eval" / "fig1"
THUMB = 16
WIN = 64


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(name: str, array: np.ndarray, lines: list[str]) -> None:
    array = np.ascontiguousarray(array)
    text = base64.b64encode(zlib.compress(array.tobytes(), 9)).decode("ascii")
    lines.append(f"ARRAY {name} {array.dtype} {','.join(str(s) for s in array.shape)} {len(text)}")
    for start in range(0, len(text), 3000):
        lines.append(text[start:start + 3000])
    lines.append("END")


def stretch_limits(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for band in range(3):
        values = stack[..., band]
        values = values[np.isfinite(values)]
        a, b = np.percentile(values, [2, 98])
        lo.append(float(a))
        hi.append(float(b))
    return np.array(lo), np.array(hi)


def apply_stretch(stack: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    scaled = (np.nan_to_num(stack, nan=0.0) - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def read_marida(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        rgb = np.stack([src.read(4), src.read(3), src.read(2)], axis=-1).astype(np.float64)
        h, w = src.shape
        cx, cy = src.xy(h // 2, w // 2)
        lon, lat = rio_transform(src.crs, "EPSG:4326", [cx], [cy])
        meta = {"crs": str(src.crs), "bounds": list(src.bounds),
                "center_latlon": [float(lat[0]), float(lon[0])]}
    return rgb, meta


def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64)


def mados_file(crop: str, token: str) -> Path:
    scene, index = crop.split("_")[1], crop.split("_")[2]
    folder = MADOS / f"Scene_{scene}" / "10"
    hits = sorted(folder.glob(f"*_{token}_{index}.tif"))
    if len(hits) != 1:
        listing = sorted(p.name for p in folder.glob("*.tif"))[:12]
        raise FileNotFoundError(f"{crop} {token}: {len(hits)} hits; folder sample {listing}")
    return hits[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    selection: dict = {"inputs": {}}
    for path in (FOLDS, DUP, INV):
        selection["inputs"][str(path.relative_to(ROOT))] = sha256(path)

    folds = read_rows(FOLDS)
    fold_of = {r["scene"]: int(r["fold"]) for r in folds}
    inventory = read_rows(INV)
    by_scene: dict[str, list[str]] = {}
    for row in inventory:
        by_scene.setdefault(row["scene"], []).append(row["stem"])

    # ---------- map: scene centroids ----------
    scenes = []
    metas: dict[str, dict] = {}
    for scene in sorted(by_scene):
        lats, lons = [], []
        for stem in sorted(by_scene[scene]):
            with rasterio.open(MARIDA / scene / f"{stem}.tif") as src:
                h, w = src.shape
                cx, cy = src.xy(h // 2, w // 2)
                lon, lat = rio_transform(src.crs, "EPSG:4326", [cx], [cy])
                metas[stem] = {"crs": str(src.crs), "bounds": list(src.bounds)}
            lats.append(lat[0])
            lons.append(lon[0])
        scenes.append({"scene": scene, "fold": fold_of.get(scene), "n_patches": len(by_scene[scene]),
                       "lat": float(np.mean(lats)), "lon": float(np.mean(lons))})
    selection["scenes"] = scenes

    # ---------- (a) scene with the most MIN4 debris instances ----------
    ranked = sorted(folds, key=lambda r: (-int(r["n_debris_min4"]), -int(r["n_patches"]), r["scene"]))
    scene_a = ranked[0]["scene"]
    stems = sorted(by_scene[scene_a])
    stack = np.stack([read_marida(MARIDA / scene_a / f"{s}.tif")[0] for s in stems])
    lo, hi = stretch_limits(stack)
    scaled = apply_stretch(stack, lo, hi)
    n, h, w, _ = scaled.shape
    thumbs = scaled.reshape(n, h // THUMB, THUMB, w // THUMB, THUMB, 3).mean(axis=(2, 4))
    encode("a_thumbs", np.uint8(np.round(thumbs * 255)), lines)
    selection["a"] = {"scene": scene_a, "fold": fold_of.get(scene_a),
                      "n_debris_min4": int(ranked[0]["n_debris_min4"]), "stems": stems,
                      "patches": [{"stem": s, **metas[s]} for s in stems],
                      "stretch_lo": lo.tolist(), "stretch_hi": hi.tolist(),
                      "thumb_px": THUMB, "patch_px": [h, w]}

    # ---------- (c) MADOS test crop duplicated from a MARIDA training patch ----------
    dup = [r for r in read_rows(DUP)
           if r["same_acq"] == "True" and r["mados_split"] == "test" and r["marida_split"] == "train"]
    dup.sort(key=lambda r: (-float(r["px_intersection"] or 0), -float(r["score"] or 0),
                            r["mados_id"], r["marida_patch"]))
    pair = dup[0]
    crop, stem, scene_c = pair["mados_id"], pair["marida_patch"], pair["marida_scene"]
    dx, dy = int(float(pair["dx"])), int(float(pair["dy"]))
    m_rgb, m_meta = read_marida(MARIDA / scene_c / f"{stem}.tif")
    m_cl = read_band(MARIDA / scene_c / f"{stem}_cl.tif")
    green = "rhorc_560"
    try:
        mados_file(crop, green)
    except FileNotFoundError:
        green = "rhorc_559"
    d_rgb = np.stack([read_band(mados_file(crop, t)) for t in ("rhorc_665", green, "rhorc_492")], axis=-1)
    d_cl = read_band(mados_file(crop, "cl"))
    H, W = m_cl.shape
    h2, w2 = d_cl.shape

    def mapped(sign: int):
        """MARIDA (y, x) <- MADOS (y - sign*dy, x - sign*dx); returns overlap slices."""
        oy, ox = sign * dy, sign * dx
        y0, y1 = max(0, oy), min(H, h2 + oy)
        x0, x1 = max(0, ox), min(W, w2 + ox)
        return (slice(y0, y1), slice(x0, x1)), (slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox))

    best = None
    sign_log = {}
    for sign in (1, -1):
        (my, mx), (dy_, dx_) = mapped(sign)
        a = m_rgb[my, mx, 0].ravel()
        b = d_rgb[dy_, dx_, 0].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 100 else float("nan")
        sign_log[str(sign)] = {"b4_r": r, "n_px": int(ok.sum())}
        if np.isfinite(r) and (best is None or r > best[0]):
            best = (r, sign)
    sign = best[1]
    (my, mx), (dy_, dx_) = mapped(sign)
    debris = np.argwhere(m_cl[my, mx] == 1)
    cy = int(round(debris[:, 0].mean())) + my.start
    cx = int(round(debris[:, 1].mean())) + mx.start
    half = WIN // 2
    cy = min(max(cy, my.start + half), my.stop - half)
    cx = min(max(cx, mx.start + half), mx.stop - half)
    wy, wx = slice(cy - half, cy + half), slice(cx - half, cx + half)
    oy, ox = sign * dy, sign * dx
    vy, vx = slice(wy.start - oy, wy.stop - oy), slice(wx.start - ox, wx.stop - ox)
    m_win, d_win = m_rgb[wy, wx], d_rgb[vy, vx]
    lo_c, hi_c = stretch_limits(np.concatenate([m_win.reshape(-1, 3), d_win.reshape(-1, 3)])[None])
    encode("c_marida_rgb", np.uint8(np.round(apply_stretch(m_win, lo_c, hi_c) * 255)), lines)
    encode("c_mados_rgb", np.uint8(np.round(apply_stretch(d_win, lo_c, hi_c) * 255)), lines)
    encode("c_marida_debris", np.uint8(m_cl[wy, wx] == 1), lines)
    encode("c_mados_debris", np.uint8(d_cl[vy, vx] == 1), lines)
    selection["c"] = {"mados_id": crop, "marida_patch": stem, "marida_scene": scene_c,
                      "dup_row": {k: pair[k] for k in ("score", "dx", "dy", "px_intersection", "px_union",
                                                        "px_mados", "px_marida", "same_acq")},
                      "offset_sign": sign, "sign_check": sign_log,
                      "marida_window_rc": [wy.start, wx.start, WIN], "mados_window_rc": [vy.start, vx.start, WIN],
                      "window_debris_px": [int((m_cl[wy, wx] == 1).sum()), int((d_cl[vy, vx] == 1).sum())],
                      "stretch_lo": lo_c.tolist(), "stretch_hi": hi_c.tolist(),
                      "n_candidates": len(dup), "marida_center_latlon": m_meta["center_latlon"]}

    (OUT / "fig1_extract.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    (OUT / "fig1_selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    print("scene_a", scene_a, "patches", len(stems))
    print("pair", crop, stem, "sign", sign, sign_log)
    print("Saved:", OUT / "fig1_extract.txt")


if __name__ == "__main__":
    main()
