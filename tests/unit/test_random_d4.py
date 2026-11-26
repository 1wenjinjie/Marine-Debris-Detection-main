from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ultralytics.data.augment import Instances

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset, RandomD4  # noqa: E402


class RandomD4Tests(unittest.TestCase):
    def test_all_8_symmetries_exact_bbox_and_lossless_pixels(self):
        s = 256
        x0, y0, bw, bh = 37, 81, 12, 6
        cx, cy = x0 + bw / 2.0, y0 + bh / 2.0
        for ch in (6, 7):
            base_img = np.zeros((s, s, ch), dtype=np.float32)
            base_img[y0:y0 + bh, x0:x0 + bw, :] = np.arange(1, ch + 1, dtype=np.float32)
            for normalized in (False, True):
                scale = float(s) if normalized else 1.0
                box = np.array([[cx / scale, cy / scale, bw / scale, bh / scale]], dtype=np.float32)
                for k in range(4):
                    for flip_lr in (False, True):
                        lbl = {
                            "img": base_img.copy(),
                            "instances": Instances(box.copy(), bbox_format="xywh", normalized=normalized),
                        }
                        out = RandomD4.apply(lbl, k=k, flip_lr=flip_lr)
                        im = out["img"]
                        ys, xs = np.where(im[..., 0] > 0)
                        gt_xyxy = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=np.float32)
                        inst = out["instances"]
                        inst.convert_bbox(format="xyxy")
                        pred_xyxy = inst.bboxes * (float(s) if normalized else 1.0)
                        self.assertEqual(float(np.abs(pred_xyxy - gt_xyxy).max()), 0.0)
                        for c in range(ch):
                            self.assertEqual(set(np.unique(im[..., c]).tolist()), {0.0, float(c + 1)})

    def test_nonsquare_180_and_flip(self):
        h, w = 128, 256
        x0, y0, bw, bh = 30, 20, 10, 8
        base_img = np.zeros((h, w, 6), dtype=np.float32)
        base_img[y0:y0 + bh, x0:x0 + bw, :] = 1.0
        box = np.array([[(x0 + bw / 2.0) / w, (y0 + bh / 2.0) / h, bw / w, bh / h]], dtype=np.float32)
        for k in (0, 2):
            for flip_lr in (False, True):
                lbl = {
                    "img": base_img.copy(),
                    "instances": Instances(box.copy(), bbox_format="xywh", normalized=True),
                }
                out = RandomD4.apply(lbl, k=k, flip_lr=flip_lr)
                ys, xs = np.where(out["img"][..., 0] > 0)
                gt_xyxy = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=np.float32)
                inst = out["instances"]
                inst.convert_bbox(format="xyxy")
                pred_xyxy = inst.bboxes * np.array([[w, h, w, h]], dtype=np.float32)
                self.assertAlmostEqual(float(np.abs(pred_xyxy - gt_xyxy).max()), 0.0, places=5)

    def test_build_transforms_toggles_d4_by_augment_and_hyp(self):
        dummy = MaskRescueNPYDataset.__new__(MaskRescueNPYDataset)
        dummy.imgsz = 512
        dummy.rect = False
        dummy.use_segments = False
        dummy.use_keypoints = False

        hyp_on = SimpleNamespace(copy_paste=0.0, degrees=180.0, flipud=0.5, fliplr=0.5)
        hyp_off = SimpleNamespace(copy_paste=0.0, degrees=0.0, flipud=0.0, fliplr=0.0)

        dummy.augment = True
        ops_train = dummy.build_transforms(hyp_on).transforms
        self.assertTrue(any(isinstance(op, RandomD4) for op in ops_train))

        ops_train_off = dummy.build_transforms(hyp_off).transforms
        self.assertFalse(any(isinstance(op, RandomD4) for op in ops_train_off))

        dummy.augment = False
        ops_val = dummy.build_transforms(hyp_on).transforms
        self.assertFalse(any(isinstance(op, RandomD4) for op in ops_val))


if __name__ == "__main__":
    unittest.main()
