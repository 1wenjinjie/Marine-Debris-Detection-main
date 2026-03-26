"""
Expand YOLO detection backbones to accept more input channels while reusing pretrained weights.

Usage example:
    python prepare_multichannel_yolo_weights.py \
        --source yolo11n.pt \
        --output yolo11n_6ch_init.pt \
        --channels 6 \
        --fill-mode mean

The script loads the pretrained checkpoint with Ultralytics YOLO, replaces the first
convolution to match the requested number of channels, and initializes the additional
channels according to the chosen strategy:
    - mean:   average the original RGB filters and copy the mean into each new channel
    - repeat: cycle through the original channels (RGBRGB...) for the new bands
    - zeros:  initialize the new channels with zeros (useful for debugging only)

The modified checkpoint keeps the original weights for the remaining layers, updates
metadata so that Ultralytics recognises the new channel count, and writes the result
to ``--output``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

FILL_MODES = {"mean", "repeat", "zeros"}


def expand_conv_weight(weight: torch.Tensor, target_channels: int, mode: str) -> torch.Tensor:
    """Expand a convolution weight tensor from (out, in, k, k) to the requested input channels."""
    if weight.ndim != 4:
        raise ValueError("Expected a 4D weight tensor (out, in, k, k)")

    out_channels, in_channels, k1, k2 = weight.shape
    if target_channels == in_channels:
        return weight.clone()
    if target_channels < in_channels:
        return weight[:, :target_channels].clone()

    new_weight = weight.new_empty(out_channels, target_channels, k1, k2)
    new_weight[:, :in_channels] = weight
    extra = target_channels - in_channels

    if mode == "mean":
        filler = weight.mean(dim=1, keepdim=True)
        new_weight[:, in_channels:] = filler.repeat(1, extra, 1, 1)
    elif mode == "repeat":
        for idx in range(extra):
            new_weight[:, in_channels + idx] = weight[:, idx % in_channels]
    elif mode == "zeros":
        new_weight[:, in_channels:] = 0
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unsupported fill mode: {mode}")

    return new_weight


def prepare_weights(source: Path, output: Path, channels: int, mode: str) -> None:
    """Load pretrained YOLO weights, expand the first conv, and save the new checkpoint."""
    model = YOLO(str(source))
    conv_module = model.model.model[0]
    old_conv = conv_module.conv

    new_conv = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )

    with torch.no_grad():
        new_conv.weight.copy_(expand_conv_weight(old_conv.weight, channels, mode))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    conv_module.conv = new_conv

    # Update metadata so Ultralytics rebuilds the model with the correct input depth.
    model.model.yaml["ch"] = channels
    model.overrides["ch"] = channels
    if hasattr(model.model, "args"):
        model.model.args["ch"] = channels

    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))

    print(f"✅ Saved multi-channel weights to: {output}")
    print(f"   - Source weights:   {source}")
    print(f"   - Input channels:   {channels}")
    print(f"   - Fill mode:        {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create multi-channel YOLO weights")
    parser.add_argument("--source", type=Path, required=True, help="Pretrained YOLO weight file (e.g. yolo11n.pt)")
    parser.add_argument("--output", type=Path, required=True, help="Destination path for the expanded checkpoint")
    parser.add_argument("--channels", type=int, default=6, help="Target number of input channels (default: 6)")
    parser.add_argument(
        "--fill-mode",
        type=str,
        default="mean",
        choices=sorted(FILL_MODES),
        help="How to initialise the additional channels (default: mean)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise SystemExit(f"❌ Source weight file not found: {args.source}")

    if args.channels < 1:
        raise SystemExit("❌ --channels 必须大于等于 1")

    prepare_weights(args.source, args.output, args.channels, args.fill_mode)


if __name__ == "__main__":
    main()
