#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import Path
import sys
import zipfile
from typing import Sequence, Set


def load_entries(path: Path) -> Set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as zf:
        return set(zf.namelist())


def require_any(entries: Set[str], candidates: Sequence[str], label: str) -> None:
    if not any(candidate in entries for candidate in candidates):
        raise RuntimeError(f"missing {label}: expected one of {', '.join(candidates)}")


def main(argv: Sequence[str]) -> int:
    if len(argv) != 3:
        print("usage: check_release_archives.py <visiong_cpp.zip> <visiong_python.zip>", file=sys.stderr)
        return 1

    cpp_zip = Path(argv[1]).resolve()
    py_zip = Path(argv[2]).resolve()

    cpp_entries = load_entries(cpp_zip)
    py_entries = load_entries(py_zip)

    require_any(cpp_entries, ["VISIONG_COMPONENTS.txt"], "cpp manifest")
    require_any(cpp_entries, ["cmake/VisionGConfig.cmake"], "cpp CMake config")
    require_any(cpp_entries, ["lib/libvisiong.so", "lib/libvisiong.a"], "cpp library")
    if not any(entry.startswith("include/visiong/") for entry in cpp_entries):
        raise RuntimeError("missing public C++ headers under include/visiong/")

    require_any(py_entries, ["VISIONG_COMPONENTS.txt"], "python manifest")
    require_any(py_entries, ["_visiong.so"], "python extension module")
    require_any(py_entries, ["visiong.py"], "python shim")

    print(f"release archives look good: {cpp_zip.name}, {py_zip.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
