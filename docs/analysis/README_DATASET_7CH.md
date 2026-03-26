# MARIDA `mask_bbox_rescue_split` — 7-channel pipeline

We now have a reproducible way to turn the curated MARIDA rescue split into
`(H, W, 7)` numpy tensors: six Sentinel-2 bands (R, G, B, NIR, SWIR1, SWIR2)
plus a KMNet-inspired knowledge heatmap derived from the original semantic
masks.

## Knowledge heatmap recipe

- Positive classes: Marine Debris, Dense/Sparse Sargassum, Natural Organic,
	Ship, Foam, and Wakes (class IDs 1,2,3,4,5,9,14).
- Each class contributes a weighted binary mask (weights skewed toward debris
	and ships) that is morphologically closed and blurred (`sigma=2.5`).
- A proximity prior is computed via `cv2.distanceTransform` on the dilated
	positives to softly highlight near-miss regions, echoing KMNet's
	knowledge-guided attention.
- The final heatmap is the normalized blend `0.75 * smoothed + 0.25 *
	proximity` in `[0, 1]` and is concatenated as channel index 6.

## Generate 7-channel caches

Run the preprocessing script once per dataset refresh (takes <1 minute for 514
patches):

```powershell
D:/OceanEdge_Group/Text/miniconda3/envs/yolo_test/python.exe preprocess_generate_knowledge_channel.py
```

This walks the manifests under `mask_bbox_rescue_split`, builds the knowledge
map for every `.npy`, and rewrites the files in place. A JSON summary is stored
at `mask_bbox_rescue_split/knowledge_generation_summary.json`.

## Validate the output

1. Rebuild the per-file histogram:

	 ```powershell
	 $env:PYTHONPATH='.'; D:/OceanEdge_Group/Text/miniconda3/envs/yolo_test/python.exe scripts/generate_channel_report.py
	 ```

	 Expect `514` files with exactly `7` channels.

2. Load a sample via the helper:

	 ```powershell
	 $env:PYTHONPATH='.'; D:/OceanEdge_Group/Text/miniconda3/envs/yolo_test/python.exe scripts/test_load_manifest.py --manifest mask_bbox_rescue_split/train.txt
	 ```

	 Output should report `image shape: (256, 256, 7)`.

## Dataset helpers recap

- `datasets/multispectral_yolo_dataset.py` — IO helper + optional PyTorch Dataset.
- `scripts/test_load_manifest.py` — single-sample sanity loader.
- `scripts/generate_channel_report.py` — channel histogram + per-file audit.
- `marida_mask_rescue_7ch.yaml` — data config for training.

Next follow-ups: plug the dataset into the dedicated trainer, add
Albumentations, and spin up a 7-channel sanity experiment.
