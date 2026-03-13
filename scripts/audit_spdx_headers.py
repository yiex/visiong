#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Audit SPDX license identifiers on project-owned files."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "LGPL-3.0-or-later"
DIRECTORIES = [
    ROOT / "src",
    ROOT / "include",
    ROOT / "scripts",
    ROOT / "docs",
    ROOT / "cmake",
    ROOT / ".github/workflows",
]
EXTRA_FILES = [
    ROOT / "README.md",
    ROOT / "THIRD_PARTY_NOTICES.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CMakeLists.txt",
    ROOT / "build.sh",
]
ALLOWED_SUFFIXES = {".c", ".cpp", ".h", ".hpp", ".py", ".sh", ".md", ".yml", ".yaml", ".cmake"}


def project_files() -> list[Path]:
    files: list[Path] = []
    for directory in DIRECTORIES:
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES:
                    files.append(path)
    for path in EXTRA_FILES:
        if path.is_file():
            files.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path not in seen and not path.relative_to(ROOT).as_posix().startswith("3rdparty/"):
            seen.add(path)
            unique.append(path)
    return sorted(unique)


def main() -> int:
    problems: list[str] = []
    checked = 0

    for path in project_files():
        checked += 1
        rel = path.relative_to(ROOT)
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:6]
        tag_lines = [line for line in lines if "SPDX-License-Identifier:" in line]
        if not tag_lines:
            problems.append(f"missing SPDX header: {rel}")
            continue
        if not any(EXPECTED in line for line in tag_lines):
            problems.append(f"unexpected SPDX identifier in {rel}: {' | '.join(tag_lines)}")

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1

    print(f"[OK] SPDX headers verified on {checked} project-owned files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

