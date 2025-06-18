"""Compat package forwarding imports from repository root."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent / "__init__.py"
_parent = _root.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
_spec = spec_from_file_location("ndml_root", _root, submodule_search_locations=[str(_root.parent)])
_module = module_from_spec(_spec)
sys.modules["ndml_root"] = _module
_spec.loader.exec_module(_module)  # type: ignore

for _name, _val in _module.__dict__.items():
    if not _name.startswith("_"):
        globals()[_name] = _val
