"""
Full teacher-task model smoke test: CBAM + BiFPN-lite + EIoU.

NOT a real training run. Trains 1 epoch, batch=4, imgsz=512, workers=0.

Usage:
  KMP_DUPLICATE_LIB_OK=TRUE python scripts/v2_full_smoke_test.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from training.mask_rescue_hyperparams import MASK_RESCUE_TRAIN_DEFAULTS
from ultralytics_plugins.bifpn_modules import install_bifpn_patch
from ultralytics_plugins.eiou_trainer import EIoUTrainer

install_bifpn_patch()

DATA_6CH = str(ROOT / "configs" / "datasets" / "marida_v2_6ch.yaml")
MODEL_FULL = str(ROOT / "configs" / "models" / "yolov11n_v2_6ch_full.yaml")
RUNS_DIR = ROOT / "runs_v2"
RUN_NAME = "smoke_full_v2"


def main() -> int:
    print("=" * 60)
    print("Full v2 smoke test")
    print(f"  model:   {MODEL_FULL}")
    print(f"  data:    {DATA_6CH}")
    print(f"  output:  {RUNS_DIR / RUN_NAME}")
    print("=" * 60)

    overrides = {
        **MASK_RESCUE_TRAIN_DEFAULTS,
        "model": MODEL_FULL,
        "data": DATA_6CH,
        "project": str(RUNS_DIR),
        "name": RUN_NAME,
        "seed": 0,
        "exist_ok": True,
        "device": "0",
        "epochs": 1,
        "batch": 4,
        "imgsz": 512,
        "workers": 0,
        "patience": 0,
        "save_period": -1,
        "plots": False,
        "val": True,
    }

    status = {"yaml_parsed": False, "trainer_built": False, "trained": False, "evaluated": False}
    try:
        print("\n[1/4] Building trainer (yaml parse + model construction)...")
        trainer = EIoUTrainer(overrides=overrides)
        status["trainer_built"] = True
        status["yaml_parsed"] = True
        print("      OK: trainer built, yaml parsed.")

        print("\n[2/4] Running 1-epoch training...")
        trainer.train()
        status["trained"] = True
        print("      OK: 1 epoch finished.")

        print("\n[3/4] Loading best.pt and running test eval...")
        from scripts.v2_evaluate import evaluate

        best = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
        if not best.exists():
            best = RUNS_DIR / RUN_NAME / "weights" / "last.pt"
        if not best.exists():
            raise FileNotFoundError(f"No weights produced: {best}")

        metrics = evaluate(weights=str(best), data_yaml=DATA_6CH, channels=6, split="test")
        status["evaluated"] = True
        print(f"      OK: eval pipeline ran. mAP50={metrics.get('mAP50', 'NA'):.4f}")

        print("\n[4/4] Smoke test PASSED.")
        out = RUNS_DIR / RUN_NAME / "smoke_status.json"
        out.write_text(json.dumps({"status": status, "ok": True}, indent=2))
        print(f"      status -> {out}")
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("SMOKE TEST FAILED")
        print("=" * 60)
        print(f"  stage status: {status}")
        print(f"  exception:    {type(e).__name__}: {e}")
        print()
        traceback.print_exc()
        try:
            (RUNS_DIR / RUN_NAME).mkdir(parents=True, exist_ok=True)
            (RUNS_DIR / RUN_NAME / "smoke_status.json").write_text(
                json.dumps({"status": status, "ok": False, "error": str(e)}, indent=2)
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
