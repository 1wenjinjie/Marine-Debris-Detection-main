"""Custom Ultralytics DetectionTrainer that plugs in the MaskRescueNPYDataset."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler

from ultralytics.data.build import build_dataloader, InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

from .mask_rescue_dataset import MaskRescueNPYDataset


class MaskRescueTrainer(DetectionTrainer):
    """Trainer that feeds YOLO with 7-channel knowledge tensors stored as .npy."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        stride = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        dataset_root = Path(self.data.get("path", "mask_bbox_rescue_split")).resolve()
        channels = self.data.get("channels", 3)
        return MaskRescueNPYDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch or self.args.batch,
            augment=mode == "train",
            hyp=self.args,
            # MARIDA patches are square. Keep validation at the locked imgsz so
            # AP evaluation and per-patch operating-point inference share the
            # same geometry instead of rect-batch-dependent 544 px padding.
            rect=self.args.rect if mode == "train" else False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            dataset_root=dataset_root,
            channels=channels,
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        if mode == "train" and self._should_use_weighted_sampler(dataset, rank):
            LOGGER.info(f"{self.args.task}: enabling confidence-weighted sampler for {len(dataset)} samples")
            return self._build_weighted_dataloader(dataset, batch_size)

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

    def _should_use_weighted_sampler(self, dataset, rank: int) -> bool:
        weights = getattr(dataset, "sample_weights", None)
        if not weights:
            return False
        if len(weights) != len(dataset):
            LOGGER.warning(f"{self.args.task}: confidence weights misaligned (len != dataset); falling back to shuffle")
            return False
        if rank not in (-1, 0):
            LOGGER.warning(f"{self.args.task}: distributed ranks {rank} not yet supported for weighted sampling")
            return False
        if not any(w > 0 for w in weights):
            LOGGER.warning(f"{self.args.task}: all confidence weights are zero; falling back to shuffle")
            return False
        return True

    def _build_weighted_dataloader(self, dataset, batch_size: int):
        batch = min(batch_size, len(dataset))
        nd = torch.cuda.device_count()
        workers = min(os.cpu_count() // max(nd, 1), self.args.workers)
        weights = torch.as_tensor(dataset.sample_weights, dtype=torch.double)
        stats = getattr(dataset, "sample_weight_stats", None)
        examples = getattr(dataset, "sample_weight_examples", None)
        if stats:
            LOGGER.info(
                f"{self.args.task}: sample weight stats -> min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}, median={stats['median']:.4f}"
            )
        if examples:
            lowest = ", ".join(f"{name}:{weight:.4f}" for name, weight in examples.get("lowest", []))
            highest = ", ".join(f"{name}:{weight:.4f}" for name, weight in examples.get("highest", []))
            LOGGER.info(f"{self.args.task}: lowest weights [{lowest}] | highest weights [{highest}]")
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=False,
            num_workers=workers,
            sampler=sampler,
            prefetch_factor=4 if workers > 0 else None,
            pin_memory=nd > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=self.args.compile and len(dataset) % max(batch, 1) != 0,
        )
