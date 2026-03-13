# SPDX-License-Identifier: LGPL-3.0-or-later
import ctypes
import importlib
import os
import sys

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_RTLD_NOW = getattr(os, "RTLD_NOW", 2)
_RTLD_GLOBAL = getattr(os, "RTLD_GLOBAL", 0x100)
_RTLD_MODE = _RTLD_NOW | _RTLD_GLOBAL
_LD_MARKER = "_VISIONG_LD_PATH_READY"


def _ensure_loader_library_path():
    required_paths = [_MODULE_DIR, "/oem/usr/lib"]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [p for p in current.split(":") if p]
    missing = [p for p in required_paths if p not in entries]

    if not missing:
        os.environ[_LD_MARKER] = "1"
        return

    if os.environ.get(_LD_MARKER) == "1":
        return

    os.environ[_LD_MARKER] = "1"
    os.environ["LD_LIBRARY_PATH"] = ":".join(required_paths + entries)

    exe = sys.executable
    if exe:
        argv = [exe] + (sys.argv if sys.argv else [])
        os.execv(exe, argv)


def _preload_rockit_global():
    candidates = (
        os.path.join(_MODULE_DIR, "librockit.so"),
        "/oem/usr/lib/librockit.so",
        os.path.join(_MODULE_DIR, "librockit_full.so"),
        "/oem/usr/lib/librockit_full.so",
    )
    for lib in candidates:
        try:
            ctypes.CDLL(lib, mode=_RTLD_MODE)
            return
        except OSError:
            continue


_ensure_loader_library_path()

_old_flags = sys.getdlopenflags()
try:
    if _MODULE_DIR not in sys.path:
        sys.path.insert(0, _MODULE_DIR)
    sys.setdlopenflags(_RTLD_MODE)
    _preload_rockit_global()
    _mod = importlib.import_module("_visiong")
finally:
    try:
        sys.setdlopenflags(_old_flags)
    except Exception:
        pass

for _name in dir(_mod):
    if _name.startswith("__") and _name not in ("__doc__", "__version__"):
        continue
    globals()[_name] = getattr(_mod, _name)

__all__ = [n for n in globals() if not n.startswith("_")]

