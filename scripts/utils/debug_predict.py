from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    model = YOLO("runs/sanity/yolo11n_6ch_clean_mosaic05/weights/best.pt")
    results = model.predict(
        source="MARIDA_YOLO_6band_split/val/images",
        imgsz=256,
        device=0,
        verbose=False,
        max_det=50,
        conf=0.001,
        stream=False,
    )
    total_boxes = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
    print(f"预测图像数量: {len(results)}")
    print(f"预测框合计: {total_boxes}")
    if results:
        first = results[0]
        print(f"首张图像: {Path(first.path).name if first.path else '未知'}")
        if first.boxes is None or len(first.boxes) == 0:
            print("首张图像没有产生检测框")
        else:
            scores = first.boxes.conf.tolist()
            print(f"首张图像置信度: {scores[:10]}")


if __name__ == "__main__":
    main()
