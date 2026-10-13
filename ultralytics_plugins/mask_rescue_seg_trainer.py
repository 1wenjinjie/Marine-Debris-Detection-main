"""SegmentationTrainer variant that uses MaskRescueSegNPYDataset (multi-channel npy)."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler

from ultralytics.data.build import build_dataloader, InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

from .mask_rescue_seg_dataset import MaskRescueSegNPYDataset


class MaskRescueSegTrainer(SegmentationTrainer):
    """Segmentation trainer wired to MaskRescueSegNPYDataset (npy multi-channel inputs)."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        stride = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        dataset_root = Path(self.data.get("path", "marida_v2_seg")).resolve()
        channels = self.data.get("channels", 3)
        return MaskRescueSegNPYDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch or self.args.batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect if mode == "train" else True,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            dataset_root=dataset_root,
            channels=channels,
            task="segment",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False

        worker_count = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=worker_count,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )
