"""Content fingerprints for MPB dataset variants."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import MutableMapping

import yaml


HashCache = MutableMapping[tuple[object, ...], str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_cache_key(path: Path) -> tuple[object, ...]:
    stat = path.stat()
    inode = getattr(stat, "st_ino", 0)
    identity: object = (stat.st_dev, inode) if inode else str(path.resolve())
    return identity, stat.st_size, stat.st_mtime_ns


def _cached_sha256(path: Path, cache: HashCache) -> str:
    key = _file_cache_key(path)
    if key not in cache:
        cache[key] = sha256_file(path)
    return cache[key]


def _resolve_manifest(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for folder in ("images", "images_7ch"):
        if folder in parts:
            parts[parts.index(folder)] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"Cannot derive label path from {image_path}")


def dataset_fingerprint(
    config_path: Path,
    hash_cache: HashCache | None = None,
) -> dict[str, object]:
    """Hash every model-visible artifact in a generated dataset variant."""
    config_path = config_path.resolve()
    cache: HashCache = hash_cache if hash_cache is not None else {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(str(data["path"])).resolve()
    channels = int(data["channels"])
    stats_path = root / f"channel_stats_{channels}ch.json"
    required = (config_path, stats_path)
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"Incomplete dataset contract: {required}")

    digest = hashlib.sha256()
    digest.update(b"mpb-dataset-fingerprint-v1\0")
    config_hash = _cached_sha256(config_path, cache)
    stats_hash = _cached_sha256(stats_path, cache)
    digest.update(f"config\0{config_hash}\0stats\0{stats_hash}\0".encode())

    manifest_hashes: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    logical_files = 2
    logical_bytes = config_path.stat().st_size + stats_path.stat().st_size
    seen_images: set[Path] = set()

    for split in ("train", "val", "test"):
        manifest = _resolve_manifest(root, data[split]).resolve()
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        manifest_hash = _cached_sha256(manifest, cache)
        manifest_hashes[split] = manifest_hash
        digest.update(f"manifest\0{split}\0{manifest_hash}\0".encode())
        logical_files += 1
        logical_bytes += manifest.stat().st_size

        image_paths = [
            Path(line.strip()).resolve()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        split_counts[split] = len(image_paths)
        for image_path in image_paths:
            label_path = _label_path(image_path).resolve()
            if image_path in seen_images:
                raise RuntimeError(f"Exact image overlap across splits: {image_path}")
            seen_images.add(image_path)
            if not image_path.exists() or not label_path.exists():
                raise FileNotFoundError(f"Missing image/label pair: {image_path}, {label_path}")
            image_hash = _cached_sha256(image_path, cache)
            label_hash = _cached_sha256(label_path, cache)
            digest.update(
                f"sample\0{split}\0{image_path.stem}\0{image_hash}\0{label_hash}\0".encode()
            )
            logical_files += 2
            logical_bytes += image_path.stat().st_size + label_path.stat().st_size

    return {
        "version": 1,
        "sha256": digest.hexdigest(),
        "config_sha256": config_hash,
        "stats_sha256": stats_hash,
        "manifest_sha256": manifest_hashes,
        "split_counts": split_counts,
        "logical_files": logical_files,
        "logical_bytes": logical_bytes,
    }
