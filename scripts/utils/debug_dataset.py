from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset


def main() -> None:
    cfg = get_cfg()
    cfg.data = "marida_6band_v11.yaml"
    cfg.imgsz = 256
    cfg.batch = 4
    data = check_det_dataset(cfg.data)
    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=data["train"],
        batch=cfg.batch,
        data=data,
        mode="train",
        stride=32,
    )
    sample = dataset[0]
    images = sample["img"]
    bboxes = sample["bboxes"]
    labels = sample["cls"]
    print(f"图像张量形状: {images.shape}")
    print(f"目标框数量: {len(bboxes)}")
    print(f"类别 ID: {labels[:10].tolist()}")


if __name__ == "__main__":
    main()
