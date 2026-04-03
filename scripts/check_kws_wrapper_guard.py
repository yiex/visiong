#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CMAKE_FILE = REPO_ROOT / "CMakeLists.txt"
BINDINGS_HEADER = REPO_ROOT / "src" / "python" / "internal" / "bindings_common.h"
BINDINGS_MODULE = REPO_ROOT / "src" / "python" / "bind_io_npu_gui.cpp"
KWS_HEADER = REPO_ROOT / "include" / "visiong" / "npu" / "KWS.h"
KWS_SOURCE = REPO_ROOT / "src" / "npu" / "KWS.cpp"


def require_text(path: Path, snippets) -> bool:
    text = path.read_text(encoding="utf-8")
    for snippet in snippets:
        if snippet not in text:
            print(f"missing '{snippet}' in {path}", file=sys.stderr)
            return False
    return True


def main() -> int:
    required_files = [KWS_HEADER, KWS_SOURCE]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        print("missing KWS wrapper files:", file=sys.stderr)
        for path in missing_files:
            print(path, file=sys.stderr)
        return 1

    ok = True
    ok &= require_text(CMAKE_FILE, ["src/npu/KWS.cpp"])
    ok &= require_text(BINDINGS_HEADER, ['#include "visiong/npu/KWS.h"'])
    ok &= require_text(BINDINGS_MODULE, ['py::class_<KWS>(m, "KWS")', 'py::class_<KWSResult>(m, "KWSResult"'])
    ok &= require_text(KWS_SOURCE, ['throw std::runtime_error("KWS: expected a model with exactly 1 input and 1 output.");'])
    if not ok:
        return 1

    print("KWS wrapper guard looks correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
