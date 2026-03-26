"""Generate train/val/test manifest txt files for mask_bbox_rescue_split."""
from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path("mask_bbox_rescue_split").resolve()
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    for split in ("train", "val", "test"):
        images = root / split / "images"
        if not images.exists():
            raise SystemExit(f"Images folder missing for split '{split}': {images}")
        files = sorted(images.glob("*.npy"))
        relative = [f.relative_to(root).as_posix() for f in files]
        manifest = root / f"{split}.txt"
        manifest.write_text("\n".join(relative))
        print(f"{split}: {len(relative)} files -> {manifest}")


if __name__ == "__main__":
    main()
