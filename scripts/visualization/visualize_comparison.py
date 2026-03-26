import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import glob
import random

def load_npy_image(path, channels=3):
    """
    Load .npy file and preprocess exactly like the training loader.
    Returns:
        img_display: (H, W, 3) uint8 BGR image for visualization
        img_tensor: (1, C, 640, 640) float32 tensor for model input
    """
    # Load raw data
    try:
        arr = np.load(path)
    except:
        return None, None

    # Handle dimensions
    if arr.ndim != 3:
        return None, None
    
    # Transpose if needed (C, H, W) -> (H, W, C)
    if arr.shape[0] in (6, 7, 11, 12) and arr.shape[0] < arr.shape[2]:
        arr = arr.transpose(1, 2, 0)
        
    # Select bands
    # 6ch: Red(B4), Green(B3), Blue(B2), NIR(B8), SWIR1(B11), SWIR2(B12) -> Indices [3, 2, 1, 7, 9, 10]
    # 3ch: Red(B4), Green(B3), Blue(B2) -> Indices [3, 2, 1]
    
    # Assuming input is 11-band or similar Sentinel-2 structure
    # If input is already 7-band (from previous steps), indices might differ.
    # Let's assume the standard 11-band structure based on previous context, 
    # OR check if it's the 7-band structure from `preprocess_extract_bands`.
    # Wait, `mask_bbox_rescue_split` has .npy files. Let's assume they are 11-band or 7-band.
    # The dataset loader handles this. Let's replicate the loader's logic.
    
    # Loader logic:
    # if arr.shape[-1] >= 11:
    #    if channels == 6: arr = arr[..., [3, 2, 1, 7, 9, 10]]
    #    else: arr = arr[..., [3, 2, 1]]
    # else: arr = arr[..., :channels]
    
    # Prepare 6ch version
    if arr.shape[-1] >= 11:
        arr_6ch = arr[..., [3, 2, 1, 7, 9, 10]]
        arr_3ch = arr[..., [3, 2, 1]]
    elif arr.shape[-1] == 7:
        # Assume [R, G, B, NIR, SWIR1, SWIR2, ?] based on training loader fallback
        arr_6ch = arr[..., :6]
        arr_3ch = arr[..., :3]
    else:
        # Fallback
        arr_6ch = arr[..., :6] if arr.shape[-1] >= 6 else arr
        arr_3ch = arr[..., :3]

    # Scale and Clip (x1500 -> 0-255)
    # Note: We need separate arrays for 3ch and 6ch models
    
    def preprocess(a):
        a = a * 1500.0
        a = np.clip(a, 0.0, 255.0)
        return a.astype(np.float32)

    proc_6ch = preprocess(arr_6ch)
    proc_3ch = preprocess(arr_3ch)
    
    # Create display image (RGB) - OpenCV uses BGR
    # proc_3ch is RGB. Convert to BGR for cv2
    img_display = proc_3ch.astype(np.uint8)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
    
    # Resize to 640x640 for model input
    # Use LetterBox-like resizing (pad to square) or just resize. 
    # For simplicity in this script, we'll just resize (stretch) as the training script does resize if rect=False.
    # But training usually uses LetterBox. Let's use simple resize for now to match 640x640 input.
    
    input_6ch = cv2.resize(proc_6ch, (640, 640))
    input_3ch = cv2.resize(proc_3ch, (640, 640))
    
    # Normalize 0-1 and CHW
    input_6ch = input_6ch / 255.0
    input_3ch = input_3ch / 255.0
    
    input_6ch = input_6ch.transpose(2, 0, 1) # HWC -> CHW
    input_3ch = input_3ch.transpose(2, 0, 1)
    
    input_6ch = np.expand_dims(input_6ch, axis=0) # Add batch dim
    input_3ch = np.expand_dims(input_3ch, axis=0)
    
    return img_display, torch.from_numpy(input_6ch).float(), torch.from_numpy(input_3ch).float()

def run_comparison(rgb_model_path, six_ch_model_path, test_dir, output_dir, num_samples=20):
    # Load models
    print(f"Loading RGB model: {rgb_model_path}")
    model_rgb = YOLO(rgb_model_path)
    
    print(f"Loading 6ch model: {six_ch_model_path}")
    model_6ch = YOLO(six_ch_model_path)
    
    # Get test files
    files = glob.glob(os.path.join(test_dir, "*.npy"))
    if not files:
        print("No .npy files found.")
        return
        
    print(f"Found {len(files)} test images.")
    
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Select samples (random or first N)
    # Let's try to find samples with labels if possible, but for now random is fine.
    samples = random.sample(files, min(len(files), num_samples))
    
    for i, fpath in enumerate(samples):
        fname = os.path.basename(fpath)
        print(f"Processing {i+1}/{len(samples)}: {fname}")
        
        img_display, tensor_6ch, tensor_3ch = load_npy_image(fpath)
        if img_display is None:
            continue
            
        # Resize display image to 640x640 for visualization consistency
        img_display = cv2.resize(img_display, (640, 640))
        
        # Inference RGB
        # Note: We pass the tensor directly. 
        # We need to ensure the model is on the correct device.
        device = model_rgb.device
        tensor_3ch = tensor_3ch.to(device)
        results_rgb = model_rgb(tensor_3ch, verbose=False)
        
        # Inference 6ch
        device = model_6ch.device
        tensor_6ch = tensor_6ch.to(device)
        results_6ch = model_6ch(tensor_6ch, verbose=False)
        
        # Annotate
        # RGB Result
        res_rgb = results_rgb[0]
        annotator_rgb = Annotator(img_display.copy(), line_width=2, example=str(model_rgb.names))
        if res_rgb.boxes:
            for box in res_rgb.boxes:
                b = box.xyxy[0].cpu().numpy()  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                annotator_rgb.box_label(b, model_rgb.names[int(c)], color=colors(int(c), True))
        img_rgb_pred = annotator_rgb.result()
        
        # 6ch Result
        res_6ch = results_6ch[0]
        annotator_6ch = Annotator(img_display.copy(), line_width=2, example=str(model_6ch.names))
        if res_6ch.boxes:
            for box in res_6ch.boxes:
                b = box.xyxy[0].cpu().numpy()
                c = box.cls
                annotator_6ch.box_label(b, model_6ch.names[int(c)], color=colors(int(c), True))
        img_6ch_pred = annotator_6ch.result()
        
        # Combine side-by-side
        # Add titles
        def add_title(img, text):
            # Add a black bar at top
            h, w = img.shape[:2]
            bar = np.zeros((40, w, 3), dtype=np.uint8)
            res = np.vstack([bar, img])
            cv2.putText(res, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return res

        final_rgb = add_title(img_rgb_pred, "RGB Model (3ch)")
        final_6ch = add_title(img_6ch_pred, "Multispectral Model (6ch)")
        
        combined = np.hstack([final_rgb, final_6ch])
        
        out_path = os.path.join(output_dir, f"compare_{fname}.jpg")
        cv2.imwrite(out_path, combined)

    print(f"Comparison saved to {output_dir}")

if __name__ == "__main__":
    RGB_MODEL = "marida_runs/yolo11n_rgb_full/weights/best.pt"
    SIX_CH_MODEL = "marida_runs/yolo11n_6ch_full/weights/best.pt"
    TEST_DIR = "mask_bbox_rescue_split/test/images"
    OUTPUT_DIR = "comparison_results"
    
    run_comparison(RGB_MODEL, SIX_CH_MODEL, TEST_DIR, OUTPUT_DIR)
