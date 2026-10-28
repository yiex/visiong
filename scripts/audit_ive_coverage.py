# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare public SDK IVE declarations with the library's actual source calls.
对照公开 SDK 声明与库源码中的 IVE 调用。

This is a static inventory. A source call is not evidence of correct hardware
output; use the on-board validation scripts for that separate requirement.
静态清单不证明硬件结果正确；实机正确性由独立验证脚本检查。
"""

import argparse
import json
from pathlib import Path
import re


HELPERS = {
    "RK_MPI_IVE_Init",
    "RK_MPI_IVE_Deinit",
    "RK_MPI_IVE_Query",
    "RK_MPI_IVE_CvtImageToData",
    "RK_MPI_IVE_CvtDataToImage",
    "RK_MPI_IVE_CvtImageToMemInfo",
    "RK_MPI_IVE_Pyramid_GetSize",
}

REPLACEMENTS = {
    "RK_MPI_IVE_LKOpticalFlow": {
        "interface": "lk_optical_flow_single",
        "uses": "RK_MPI_IVE_LKOpticalFlowPyr",
        "reason": "The older entry fixes max_level=3; single-level computation uses max_level=0 on the public configurable entry.",
    },
}


def remove_comments_and_strings(text):
    return re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"', " ", text, flags=re.DOTALL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "header", type=Path, help="Public rk_mpi_ive.h from the target SDK"
    )
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1] / "src"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    declarations = set(
        re.findall(
            r"\b(?:RK_S32|RK_VOID)\s+(RK_MPI_IVE_\w+)\s*\(",
            remove_comments_and_strings(args.header.read_text(encoding="utf-8")),
        )
    )
    if not declarations or not HELPERS <= declarations:
        parser.error("Header does not contain the expected public IVE declarations.")
    calls = set()
    for source in args.source.rglob("*.cpp"):
        calls.update(
            re.findall(
                r"\b(RK_MPI_IVE_\w+)\s*\(",
                remove_comments_and_strings(source.read_text(encoding="utf-8")),
            )
        )
    execution = declarations - HELPERS
    missing = execution - calls
    replaced = {
        name: entry
        for name, entry in REPLACEMENTS.items()
        if name in missing and entry["uses"] in calls
    }
    report = {
        "declared_functions": len(declarations),
        "execution_entries": len(execution),
        "called_execution_entries": sorted(execution & calls),
        "missing_execution_entries": sorted(missing),
        "replaced_execution_entries": replaced,
        "unaccounted_execution_entries": sorted(missing - replaced.keys()),
        "used_helpers": sorted(HELPERS & calls),
        "unused_helpers": sorted(HELPERS - calls),
        "undeclared_calls": sorted(calls - declarations),
        "hardware_validation": "not determined by this static inventory",
    }
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")
    raise SystemExit(1 if report["unaccounted_execution_entries"] else 0)


if __name__ == "__main__":
    main()
