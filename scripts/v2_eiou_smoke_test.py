"""
EIoU v2 smoke test.

Goal: verify the EIoU trainer patches the Ultralytics box loss and runs
forward+backward+eval end-to-end on the MARIDA v2 6ch baseline model.

NOT a real training run. Trains 1 epoch, batch=4, imgsz=512, workers=0.

Usage:
  python scripts/v2_eiou_smoke_test.py
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
from ultralytics_plugins.eiou_trainer import EIoUTrainer

DATA_6CH = str(ROOT / "configs" / "datasets" / "marida_v2_6ch.yaml")
MODEL_6CH = str(ROOT / "configs" / "models" / "yolov11n_v2_6ch.yaml")
RUNS_DIR = ROOT / "runs_v2"
RUN_NAME = "smoke_eiou_v2"


def main() -> int:
    print("=" * 60)
    print("EIoU v2 smoke test")
    print(f"  model:   {MODEL_6CH}")
    print(f"  data:    {DATA_6CH}")
    print(f"  output:  {RUNS_DIR / RUN_NAME}")
    print("=" * 60)

    overrides = {
        **MASK_RESCUE_TRAIN_DEFAULTS,
        "model": MODEL_6CH,
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

    status = {"trainer_built": False, "trained": False, "evaluated": False}
    try:
        print("\n[1/3] Building trainer...")
        trainer = EIoUTrainer(overrides=overrides)
        status["trainer_built"] = True
        print("      OK: trainer built.")

        print("\n[2/3] Running 1-epoch EIoU training...")
        trainer.train()
        status["trained"] = True
        print("      OK: 1 epoch finished.")

        print("\n[3/3] Loading weights and running test eval...")
        from scripts.v2_evaluate import evaluate

        best = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
        if not best.exists():
            best = RUNS_DIR / RUN_NAME / "weights" / "last.pt"
        if not best.exists():
            raise FileNotFoundError(f"No weights produced: {best}")

        metrics = evaluate(weights=str(best), data_yaml=DATA_6CH, channels=6, split="test")
        status["evaluated"] = True
        print(f"      OK: eval pipeline ran. mAP50={metrics.get('mAP50', 'NA'):.4f}")

        out = RUNS_DIR / RUN_NAME / "smoke_status.json"
        out.write_text(json.dumps({"status": status, "ok": True}, indent=2))
        print("\nSmoke test PASSED.")
        print(f"      status -> {out}")
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("EIOU SMOKE TEST FAILED")
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
