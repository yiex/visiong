#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_CPP = REPO_ROOT / "src" / "npu" / "internal" / "tracking" / "core.cpp"
RUN_LINE = "const nn_error_e t_ret = t_engine_->Run(t_inputs, t_output_tensors_, t_want_float_);"
CHECK_LINE = "if (t_ret != NN_SUCCESS)"


def main() -> int:
    text = CORE_CPP.read_text(encoding="utf-8")
    lines = text.splitlines()

    try:
        run_idx = lines.index(f"    {RUN_LINE}")
    except ValueError:
        print(f"missing expected NanoTrack run guard in {CORE_CPP}", file=sys.stderr)
        return 1

    expected_window = lines[run_idx : run_idx + 4]
    if f"    {CHECK_LINE}" not in expected_window:
        print(f"missing expected NanoTrack return-code check in {CORE_CPP}", file=sys.stderr)
        return 1

    broken_snippet = [
        "    std::vector<tensor_data_s> t_inputs;",
        "    t_inputs.push_back(t_input_tensor_);",
        "    {",
        '        throw std::runtime_error("NanoTrackCore::init template backbone run failed");',
    ]
    for idx in range(len(lines) - len(broken_snippet) + 1):
        if lines[idx : idx + len(broken_snippet)] == broken_snippet:
            print(f"found broken unconditional NanoTrack throw sequence in {CORE_CPP}", file=sys.stderr)
            return 1

    print("NanoTrack init guard looks correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
