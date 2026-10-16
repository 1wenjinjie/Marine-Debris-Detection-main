import sys

print("NOTION_SMOKE_OK")
print("python:", sys.version)
print("executable:", sys.executable)

try:
    import torch

    print("torch:", torch.__version__)
    available = torch.cuda.is_available()
    print("cuda_available:", available)
    if available:
        print("cuda_device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch import failed:", exc)

try:
    import ultralytics

    print("ultralytics:", ultralytics.__version__)
except Exception as exc:
    print("ultralytics import failed:", exc)

sys.exit(0)
