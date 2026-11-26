"""Register modules missing from ultralytics.nn.tasks global namespace.

Some attention/utility modules (e.g. CBAM) are defined in
ultralytics.nn.modules but are not imported at the top of
ultralytics.nn.tasks. parse_model() resolves YAML module names via that
file's globals(), so referencing those modules in a YAML raises
KeyError despite the class being importable elsewhere.

Importing this module patches ultralytics.nn.tasks.__dict__ in-place.
Idempotent: only inserts a name if it is not already present.

NOTE on c1 values in YAML:
parse_model does NOT auto-scale c1 for CBAM (it is not in the local
``base_modules`` frozenset). The c1 you write in YAML is passed through
verbatim. For scale=n (width=0.25, max_channels=1024), pre-scale your
nominal channel counts:

    c1_actual = make_divisible(min(c1_nominal, max_channels) * width, 8)

For yolov11n (scale=n):
    256 -> 64
    512 -> 128
    1024 -> 256
"""

from ultralytics.nn import tasks as _tasks
from ultralytics.nn.modules import CBAM as _CBAM

_EXTRAS = {
    "CBAM": _CBAM,
}

for _name, _cls in _EXTRAS.items():
    if _name not in _tasks.__dict__:
        _tasks.__dict__[_name] = _cls
