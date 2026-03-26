import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics_plugins.mask_rescue_dataset import MaskRescueNPYDataset

def debug_predictions():
    # 1. Load the trained model
    weights_path = Path("runs_marida/yolov11n_overfit_v3/weights/best.pt")
    if not weights_path.exists():
        print(f"Weights not found at {weights_path}")
        # Fallback to last.pt if best.pt doesn't exist
        weights_path = Path("runs_marida/yolov11n_overfit_v3/weights/last.pt")
        if not weights_path.exists():
             return
    
    print(f"Loading model from {weights_path}...")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Load validation dataset
    print("Loading validation dataset...")
    dataset_root = Path("mask_bbox_rescue_split")
    val_list = dataset_root / "val.txt"
    
    # Mock data dict for init
    mock_data = {"channels": 7, "names": ["Marine_Debris", "Dense_Sargassum", "Sparse_Sargassum", "Natural_Organic", "Ship", "Foam", "Wakes"]}
    
    ds = MaskRescueNPYDataset(
        img_path=str(val_list),
        imgsz=512,
        batch_size=1,
        augment=False,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="[ValDebug] ",
        data=mock_data
    )

    print(f"Checking first 5 validation samples...")
    
    # Inspect the first item from the dataset to see what the model gets
    try:
        item = ds[0]
        img = item['img']  # Tensor (C, H, W)
        print(f"\n--- Dataset Item 0 Inspection ---")
        print(f"Image Shape: {img.shape}")
        print(f"Image Type: {img.dtype}")
        print(f"Image Min: {img.min()}, Max: {img.max()}, Mean: {img.mean()}")
        print(f"Labels: {item['cls'].shape} classes, {item['bboxes'].shape} boxes")
        if item['bboxes'].shape[0] > 0:
            print(f"First Box: {item['bboxes'][0]}")
    except Exception as e:
        print(f"Error inspecting dataset item: {e}")

    device = next(model.model.parameters()).device
    model.model.eval()

    for i in range(5):
        # Load image and labels
        # load_image returns (H, W, C) numpy array
        im_npy, _, _ = ds.load_image(i)
        label_info = ds.get_labels()[i]
        
        im_file = label_info['im_file']
        gt_cls = label_info['cls'].flatten()
        gt_bboxes = label_info['bboxes'] # xywh normalized
        
        print(f"\n--- Sample {i}: {Path(im_file).name} ---")
        print(f"GT Boxes (xywh norm): \n{gt_bboxes}")
        print(f"GT Classes: {gt_cls}")

        # Prepare input for model
        # Model expects (B, C, H, W)
        # im_npy is (H, W, C), need to transpose to (C, H, W) and add batch dim
        im_tensor = torch.from_numpy(im_npy).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Run inference
        with torch.no_grad():
            # We use the model directly to avoid preprocessing pipeline issues with 7 channels in 'predict'
            preds = model.model(im_tensor)
            
            # --- Added Inspection ---
            raw_preds = preds[0] if isinstance(preds, (list, tuple)) else preds
            # raw_preds shape: (B, 4+Nc, Anchors)
            print(f"Raw Preds Shape: {raw_preds.shape}")
            
            # Check max confidence
            cls_preds = raw_preds[:, 4:, :]
            print(f"Max Class Score (Raw): {cls_preds.max()}")
            print(f"Min Class Score (Raw): {cls_preds.min()}")
            print(f"Mean Class Score (Raw): {cls_preds.mean()}")
            
            max_conf, _ = cls_preds.max(dim=1)
            print(f"Max Confidence across all anchors: {max_conf.max()}")
            # ------------------------
            
            # preds is usually a list, or a tuple. 
            # For detection, it returns [predictions, proto] or just predictions depending on head
            # Ultralytics v8/v11 head returns (preds, proto) during training, but during eval?
            # Let's use non_max_suppression from ultralytics.utils.ops
            
            # Fix import for non_max_suppression
            # In newer ultralytics versions, it might be in a different place or not exposed in ops
            # It is usually in ultralytics.utils.ops but maybe not exported in __init__?
            # Let's try importing from ultralytics.utils.ops directly if possible, or check where it is.
            # Actually, for v8.3+, it should be there.
            # Let's try importing from ultralytics.nn.autobackend or similar? No.
            # Let's try importing from the top level if possible, or just implement a simple one if needed.
            # Wait, the dir(ops) output didn't show it. That's strange.
            # Let's try: from ultralytics.utils import ops
            # and check if ops has it.
            # If not, maybe it's in ultralytics.utils.metrics? No.
            # It is definitely used in the codebase.
            # Let's try to use the model's own predictor logic if possible.
            # Or just use torchvision.ops.nms directly?
            # But we need the full postprocessing (conf thresh, etc).
            
            # Alternative: use model.predict() but we need to hack the input.
            # model(tensor) returns raw output.
            # Let's try to find where non_max_suppression is.
            # It might be in ultralytics.utils.ops but the dir() didn't show it?
            # Maybe it was renamed?
            # Let's try importing it from ultralytics.utils.ops explicitly again, maybe I missed it?
            # No, dir(ops) was clear.
            
            # Let's try: from ultralytics.utils.ops import non_max_suppression
            # If that fails, maybe it's in ultralytics.nn.modules.head? No.
            
            # Let's try to use the one from torchvision for a quick check if we can't find it.
            # But wait, I can just use model.predict() with a custom override?
            # No, let's try to find it.
            # In 8.3.x, it should be in ultralytics.utils.ops.
            # Maybe I am looking at a different file?
            
            # Let's try this:
            import ultralytics.utils.ops as ops
            import torchvision
            
            def simple_nms(prediction, conf_thres=0.25, iou_thres=0.45):
                # prediction is (B, 4+C, N) or (B, N, 4+C)
                # YOLOv8 output is usually (B, 4+C, 8400)
                
                bs = prediction.shape[0]
                nc = prediction.shape[1] - 4
                nm = prediction.shape[1] - nc - 4
                mi = 4 + nc
                xc = prediction[:, 4:mi].amax(1) > conf_thres
                
                output = [torch.zeros((0, 6), device=prediction.device)] * bs
                
                for xi, x in enumerate(prediction):
                    # x is (4+C, N)
                    x = x.transpose(0, -1)[xc[xi]] # (N_filtered, 4+C)
                    
                    if not x.shape[0]:
                        continue
                        
                    # Split boxes and classes
                    box, cls = x.split((4, nc), 1)
                    
                    # Box conversion (xywh to xyxy)
                    xywh = box
                    xyxy = ops.xywh2xyxy(xywh)
                    
                    # Best class
                    conf, j = cls.max(1, keepdim=True)
                    x = torch.cat((xyxy, conf, j.float()), 1)[conf.view(-1) > conf_thres]
                    
                    # NMS
                    c = x[:, 5:6] * 7680  # classes
                    boxes, scores = x[:, :4] + c, x[:, 4]
                    i = torchvision.ops.nms(boxes, scores, iou_thres)
                    output[xi] = x[i]
                    
                return output

            post_preds = simple_nms(preds[0] if isinstance(preds, (tuple, list)) else preds, conf_thres=0.001, iou_thres=0.7)
            
            det = post_preds[0] # First image in batch
            
            if det is None or len(det) == 0:
                print("PREDICTION: No detections found (even with conf=0.001).")
            else:
                # det is (N, 6) -> x1, y1, x2, y2, conf, cls
                print(f"PREDICTION: Found {len(det)} boxes (conf>0.001)")
                # Print top 5 highest confidence
                sorted_indices = torch.argsort(det[:, 4], descending=True)
                for idx in sorted_indices[:5]:
                    box = det[idx]
                    print(f"  Class: {int(box[5])} | Conf: {box[4]:.4f} | Box(xyxy): {box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}")

if __name__ == "__main__":
    debug_predictions()
