#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CMAKE_FILE = REPO_ROOT / "CMakeLists.txt"
BINDINGS_HEADER = REPO_ROOT / "src" / "python" / "internal" / "bindings_common.h"
BINDINGS_MODULE = REPO_ROOT / "src" / "python" / "bind_io_npu_gui.cpp"
ASR_HEADER = REPO_ROOT / "include" / "visiong" / "npu" / "ASR.h"
ASR_SOURCE = REPO_ROOT / "src" / "npu" / "ASR.cpp"


def require_text(path: Path, snippets) -> bool:
    text = path.read_text(encoding="utf-8")
    for snippet in snippets:
        if snippet not in text:
            print(f"missing '{snippet}' in {path}", file=sys.stderr)
            return False
    return True


def main() -> int:
    required_files = [ASR_HEADER, ASR_SOURCE]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        print("missing ASR wrapper files:", file=sys.stderr)
        for path in missing_files:
            print(path, file=sys.stderr)
        return 1

    ok = True
    ok &= require_text(CMAKE_FILE, ["src/npu/ASR.cpp"])
    ok &= require_text(BINDINGS_HEADER, ['#include "visiong/npu/ASR.h"'])
    ok &= require_text(BINDINGS_MODULE, ['py::class_<ASR>(m, "ASR")', 'py::class_<ASRResult>(m, "ASRResult"', '"char_lm_path"_a = ""', '.def_readonly("rerank_run_us", &ASRResult::rerank_run_us'])
    ok &= require_text(ASR_HEADER, ['bool has_char_lm() const;', 'int64_t last_rerank_run_us() const;', 'int64_t rerank_run_us = 0;'])
    ok &= require_text(ASR_SOURCE, ['throw std::runtime_error("ASR: acoustic model must have exactly 1 input and 1 output.");', 'struct ASR::CharNgramLm', 'm_last_rerank_run_us'])
    if not ok:
        return 1

    print("ASR wrapper guard looks correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
