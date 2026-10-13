"""v2.1 module registration. Currently empty -- v2.1 does NOT use CBAM
or BiFPN (decision: see docs/paper/v2_1/v2_1_DATASET_SPEC.md, §11 task
definition and §3 detector architecture).

If v2.1 ablations later introduce custom modules, register them here.
This function is called once on package import."""

from __future__ import annotations


def register_v2_1_modules() -> None:
    """No-op placeholder. v2.1 main + extended models use only stock
    ultralytics modules. Kept as an extension point for future ablations.
    """
    return None
