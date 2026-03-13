#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Audit a public source-release archive for high-risk content leakage."""

from __future__ import annotations

from pathlib import PurePosixPath
import sys
import zipfile

ROOT_PREFIX = "visiong"
ALLOWED_TOP_LEVEL = {
    ".github",
    ".gitattributes",
    ".gitignore",
    "3rdparty",
    "CMakeLists.txt",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "build.sh",
    "cmake",
    "docs",
    "include",
    "licenses",
    "release",
    "scripts",
    "src",
}
REQUIRED_FILES = {
    ".github/workflows/release.yml",
    "3rdparty/media-server/LICENSE",
    "3rdparty/nuklear/LICENSE",
    "3rdparty/pybind11/LICENSE",
    "3rdparty/quirc/LICENSE",
    "3rdparty/stb/LICENSE",
    "CMakeLists.txt",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "include/visiong/core/Camera.h",
    "scripts/audit_build_inputs.py",
    "scripts/audit_source_release.py",
    "scripts/audit_dependency_policy.py",
    "scripts/audit_spdx_headers.py",
    "scripts/audit_binary_release_policy.py",
    "scripts/dependency_policy.json",
    "scripts/binary_release_policy.json",
    "scripts/build_minimal_opencv.sh",
    "scripts/prepare_binary_artifact.sh",
    "scripts/create_release_archives.sh",
    "release/release_components.cmake",
    "docs/DEPENDENCY_REDISTRIBUTION_MATRIX.md",
    "docs/BINARY_REDISTRIBUTION_POLICY.md",
    "docs/BINARY_ARTIFACT_WORKFLOW.md",
    ".github/workflows/binary-artifact.yml",
    "scripts/create_source_release.sh",
    "src/core/Camera.cpp",
}
FORBIDDEN_PREFIXES = {
    ".git",
    "_deps",
    "_stage",
    "build",
    "dist",
    "examples",
    "lib",
    "opencv",
    "tests",
    "toolchain",
    "vendor",
    "3rdparty/allocator",
    "3rdparty/ive",
    "3rdparty/librga",
    "3rdparty/rknpu2",
    "3rdparty/rockchip_samples",
    "3rdparty/target_python",
}
FORBIDDEN_SUFFIXES = (
    ".a",
    ".log",
    ".o",
    ".so",
    ".tar",
    ".tar.gz",
    ".tar.xz",
    ".zip",
)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: audit_source_release.py <path-to-source-release.zip>", file=sys.stderr)
        return 1

    archive_path = sys.argv[1]
    problems: list[str] = []
    files: set[str] = set()

    with zipfile.ZipFile(archive_path) as archive:
        for raw_name in archive.namelist():
            if raw_name.endswith("/"):
                continue
            path = PurePosixPath(raw_name)
            if not path.parts or path.parts[0] != ROOT_PREFIX:
                problems.append(f"archive entry is outside expected root: {raw_name}")
                continue

            rel_path = PurePosixPath(*path.parts[1:])
            if not rel_path.parts:
                continue

            rel_text = rel_path.as_posix()
            files.add(rel_text)

            top_level = rel_path.parts[0]
            if top_level not in ALLOWED_TOP_LEVEL:
                problems.append(f"unexpected top-level entry in source archive: {rel_text}")

            for forbidden in FORBIDDEN_PREFIXES:
                if rel_text == forbidden or rel_text.startswith(forbidden + "/"):
                    problems.append(f"forbidden path leaked into source archive: {rel_text}")
                    break

            for suffix in FORBIDDEN_SUFFIXES:
                if rel_text.endswith(suffix):
                    problems.append(f"forbidden artifact type leaked into source archive: {rel_text}")
                    break

    for required in sorted(REQUIRED_FILES):
        if required not in files:
            problems.append(f"required file missing from source archive: {required}")

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1

    print("[OK] source release archive passed public-release audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
