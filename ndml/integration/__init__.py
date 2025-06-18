from importlib import import_module
from pathlib import Path
import sys

_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
_module = import_module('integration')
for _n, _v in list(_module.__dict__.items()):
    if not _n.startswith('_'):
        globals()[_n] = _v
