"""Custom entry point for KMNet-style 7-channel MARIDA training.

Highlights
---------
1. 自动权重嫁接：若不存�?7 通道检查点，自动从 COCO �?`yolo11n.pt`
   扩展�?`yolo11n_7ch_init.pt`（均值填充第 4~7 通道）�?
2. 兼容 `.npy` HWC 缓存：直接依赖我们预先生成的 7 通道 numpy�?
   不再受限�?Ultralytics 默认�?JPG 读取流程�?
3. Albumentations 同步增强：脚本会检�?albumentations 是否安装�?
   Ultralytics 内部 pipeline 会自动启�?7 通道一致增强�?
4. Sanity Check 模式：`--sanity` 仅跑 1 epoch、小批量，专注验�?
   Forward/Backward 是否正常下降，便于快速冒烟测试�?

运行示例
--------
常规训练�?0 epochs）：
    python train_7ch_main.py --epochs 50 --batch 8 --device 0

Sanity Check�? epoch）：
    python train_7ch_main.py --sanity --batch 2 --device 0
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys; sys.path.insert(0, str(Path(__file__).parent.parent / \"utils\")); from prepare_multichannel_yolo_weights import prepare_weights
from ultralytics_plugins.mask_rescue_trainer import MaskRescueTrainer

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "�?未找�?ultralytics，请�?`pip install ultralytics`�?
    ) from exc

try:
    import albumentations as _A  # noqa: F401
    ALB_AVAILABLE = True
except Exception:
    ALB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on 7-channel MARIDA dataset")
    parser.add_argument("--data-config", type=Path, default=Path("marida_mask_rescue_7ch.yaml"))
    parser.add_argument("--base-weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--weights", type=Path, default=Path("yolo11n_7ch_init.pt"))
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs_marida"))
    parser.add_argument("--name", type=str, default="yolo11n_7ch_main")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use (0-1)")
    parser.add_argument("--lr0", type=float, default=0.0025)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--close-mosaic", type=int, default=5)
    parser.add_argument("--auto-augment", type=str, default="randaugment")
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument(
        "--copy-paste-classes",
        type=str,
        default=None,
        help="Comma-separated class ids to allow for copy-paste; if unset, falls back to default behavior",
    )
    parser.add_argument("--erasing", type=float, default=0.4)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--disable-mosaic", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sanity", action="store_true", help="Run 1-epoch sanity check")
    parser.add_argument("--cache", type=str, default=None, help="Ultralytics cache mode (ram, disk, True/False)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--summary", type=Path, default=None, help="Where to dump JSON summary (defaults to run dir)")
    parser.add_argument("--save-json", action="store_true", help="Whether to dump COCO eval json")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine LR schedule")
    parser.add_argument(
        "--disable-monitor-plots",
        action="store_true",
        help="Disable live matplotlib plots in TrainingMonitor (JSON logging remains active)",
    )
    return parser.parse_args()


def ensure_weights(base: Path, target: Path, channels: int = 7) -> None:
    if target.exists():
        print(f"�?已检测到多通道权重: {target}")
        return
    if not base.exists():
        raise SystemExit(f"�?未找到基础权重 {base}，请先下�?yolo11n.pt")
    print(f"🚧 找不�?{target}，开始从 {base} 扩展 {channels} 通道权重...")
    prepare_weights(base, target, channels=channels, mode="mean")


def apply_sanity_overrides(args: argparse.Namespace) -> None:
    if not args.sanity:
        return
    print("⚙️  Sanity 模式开启：仅跑 1 epoch、小批量、限制子集�?)
    args.epochs = 1
    args.batch = min(args.batch, 2)
    args.workers = min(args.workers, 2)
    args.fraction = min(args.fraction, 0.1)
    args.patience = 0
    args.close_mosaic = 1
    args.cache = args.cache or "ram"
    if not args.name or args.name == "yolo11n_7ch_main":
        args.name = "yolo11n_7ch_sanity"


def read_results_csv(run_dir: Path) -> Dict[str, float]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    lines = [line.strip() for line in csv_path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return {}
    header = lines[0].split(",")
    last = lines[-1].split(",")
    metrics: Dict[str, float] = {}
    for key, val in zip(header, last):
        key = key.strip()
        try:
            metrics[key] = float(val)
        except ValueError:
            metrics[key] = val
    return metrics


def run_training(args: argparse.Namespace) -> Path:
    overrides: Dict[str, Any] = dict(
        data=str(args.data_config),
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        pretrained=True,
        val=not args.no_val,
        cache=args.cache,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        auto_augment=args.auto_augment,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=0.0 if args.disable_mosaic else 1.0,
        fraction=args.fraction,
        amp=not args.no_amp,
        resume=args.resume,
        seed=args.seed,
        exist_ok=args.exist_ok,
        save_json=args.save_json,
        cos_lr=args.cos_lr,
    )

    print("\n============= 训练配置 =============")
    for k in sorted(overrides):
        print(f"{k:>14}: {overrides[k]}")
    print("===================================\n")

    model = YOLO(str(args.weights))
    attach_callbacks(model, enable_plots=not args.disable_monitor_plots)
    results = model.train(trainer=MaskRescueTrainer, **overrides)
    save_dir = None
    # Prefer the trainer's actual save_dir (accounts for Ultralytics auto-increment logic)
    trainer = getattr(model, "trainer", None)
    if trainer is not None and hasattr(trainer, "save_dir"):
        save_dir = Path(trainer.save_dir)
    elif isinstance(results, dict) and "save_dir" in results:
        save_dir = Path(results["save_dir"])
    if save_dir is None:
        save_dir = Path(str(args.project)) / args.name
    return save_dir


def attach_callbacks(model, enable_plots: bool = True) -> None:
    try:
        module = importlib.import_module("callbacks.training_monitor")
        TrainingMonitorCallback = getattr(module, "TrainingMonitorCallback")
        monitor = TrainingMonitorCallback(enable_plots=enable_plots)
        model.add_callback("on_fit_epoch_end", monitor)
        model.add_callback("on_train_end", monitor.on_train_end)
        note = "with plots" if enable_plots else "JSON-only"
        print(f"📈 TrainingMonitor callback attached ({note}).")
    except Exception as exc:  # pragma: no cover - fallback when matplotlib missing
        print(f"⚠️ 无法附加 TrainingMonitor 回调: {exc}")


def dump_summary(args: argparse.Namespace, run_dir: Path, elapsed: float) -> Path:
    metrics = read_results_csv(run_dir)
    summary = {
        "run_dir": str(run_dir.resolve()),
        "weights": str(args.weights.resolve()),
        "data": str(args.data_config.resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "sanity": args.sanity,
        "elapsed_minutes": round(elapsed / 60.0, 3),
        "metrics": metrics,
    }
    summary_path = args.summary or (run_dir / "train_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"📄  结果摘要保存�?{summary_path}")
    return summary_path


def main() -> None:
    args = parse_args()
    if args.copy_paste_classes:
        args.copy_paste_classes = [int(x) for x in args.copy_paste_classes.split(",") if x.strip()]
        os.environ["MASKRESCUE_COPY_PASTE_CLASSES"] = ",".join(map(str, args.copy_paste_classes))
    args.project = args.project.resolve()
    args.data_config = args.data_config.resolve()
    args.base_weights = args.base_weights.resolve()
    args.weights = args.weights.resolve()
    apply_sanity_overrides(args)

    if not args.data_config.exists():
        raise SystemExit(f"�?找不到数据配置文�? {args.data_config}")

    if not ALB_AVAILABLE:
        print("⚠️  未检测到 albumentations，将退回到基础增强。建�?`pip install albumentations` 以获得和 KMNet 一致的几何增强�?)

    ensure_weights(args.base_weights, args.weights)

    args.project.mkdir(parents=True, exist_ok=True)
    start = time.time()
    run_dir = run_training(args)
    elapsed = time.time() - start
    dump_summary(args, run_dir, elapsed)
    print(f"�?训练完成，用�?{elapsed/60.0:.2f} 分钟。结果目�? {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("�?训练被用户中断�?)
