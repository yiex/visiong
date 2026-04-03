#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CMAKE_FILE = REPO_ROOT / "CMakeLists.txt"
BINDINGS_HEADER = REPO_ROOT / "src" / "python" / "internal" / "bindings_common.h"
BINDINGS_MODULE = REPO_ROOT / "src" / "python" / "visiong_bindings.cpp"
AUDIO_HEADER = REPO_ROOT / "include" / "visiong" / "audio" / "KwsFrontend.h"
AUDIO_SOURCE = REPO_ROOT / "src" / "audio" / "KwsFrontend.cpp"
AUDIO_BINDING = REPO_ROOT / "src" / "python" / "bind_audio.cpp"


def require_text(path: Path, snippets) -> bool:
    text = path.read_text(encoding="utf-8")
    for snippet in snippets:
        if snippet not in text:
            print(f"missing '{snippet}' in {path}", file=sys.stderr)
            return False
    return True


def main() -> int:
    required_files = [AUDIO_HEADER, AUDIO_SOURCE, AUDIO_BINDING]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        print("missing KWS audio frontend files:", file=sys.stderr)
        for path in missing_files:
            print(path, file=sys.stderr)
        return 1

    ok = True
    ok &= require_text(CMAKE_FILE, ["src/audio/KwsFrontend.cpp", "src/python/bind_audio.cpp"])
    ok &= require_text(BINDINGS_HEADER, ['#include "visiong/audio/KwsFrontend.h"', "void bind_audio(py::module_& m);"])
    ok &= require_text(BINDINGS_MODULE, ["bind_audio(m);"])
    ok &= require_text(AUDIO_BINDING, ['py::class_<KwsLogMelFrontend>(m, "KwsLogMelFrontend"'])
    ok &= require_text(
        AUDIO_SOURCE,
        [
            "std::log(energy + config_.epsilon)",
            "Match the training/reference pipeline",
        ],
    )
    if not ok:
        return 1

    print("KWS audio frontend guard looks correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
