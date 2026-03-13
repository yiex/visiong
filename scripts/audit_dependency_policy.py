#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Audit dependency redistribution policy against the repository tree."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "scripts/dependency_policy.json"
THIRD_PARTY_NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
README_FILES = [ROOT / "README.md"]
COMPLIANCE_DOC = ROOT / "docs/OPEN_SOURCE_COMPLIANCE.md"
MATRIX_DOC = ROOT / "docs/DEPENDENCY_REDISTRIBUTION_MATRIX.md"


def load_policy() -> dict:
    with POLICY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def path_is_git_tracked(path: Path) -> bool:
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    problems: list[str] = []
    notes: list[str] = []
    policy = load_policy()

    release_mode = policy.get("public_release_mode")
    if release_mode not in {"source-only", "source-and-dynamic-binaries"}:
        problems.append(
            "dependency policy must declare public_release_mode as source-only or source-and-dynamic-binaries"
        )

    notices_text = THIRD_PARTY_NOTICES.read_text(encoding="utf-8")
    readme_texts = [path.read_text(encoding="utf-8") for path in README_FILES if path.exists()]
    combined_docs = "\n".join(
        [
            notices_text,
            *readme_texts,
            COMPLIANCE_DOC.read_text(encoding="utf-8"),
            MATRIX_DOC.read_text(encoding="utf-8"),
        ]
    )
    combined_docs_lower = combined_docs.lower()
    notices_text_lower = notices_text.lower()

    expected_third_party_dirs = set()
    for entry in policy.get("in_tree_third_party", []):
        rel_path = Path(entry["path"])
        abs_path = ROOT / rel_path
        expected_third_party_dirs.add(rel_path.as_posix())
        if not abs_path.exists():
            problems.append(f"in-tree dependency path missing: {rel_path.as_posix()}")
            continue
        if entry.get("requires_local_license"):
            license_path = abs_path / "LICENSE"
            if not license_path.is_file():
                problems.append(f"in-tree dependency is missing local LICENSE: {rel_path.as_posix()}")
        if rel_path.as_posix().lower() not in notices_text_lower:
            problems.append(f"THIRD_PARTY_NOTICES.md is missing dependency path: {rel_path.as_posix()}")
        if entry["name"].lower() not in notices_text_lower:
            problems.append(f"THIRD_PARTY_NOTICES.md is missing dependency name: {entry['name']}")
        notes.append(f"in-tree ok: {entry['name']} ({rel_path.as_posix()})")

    actual_third_party_dirs = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "3rdparty").iterdir()
        if path.is_dir()
    }
    if actual_third_party_dirs != expected_third_party_dirs:
        unexpected = sorted(actual_third_party_dirs - expected_third_party_dirs)
        missing = sorted(expected_third_party_dirs - actual_third_party_dirs)
        if unexpected:
            problems.append(f"unexpected top-level 3rdparty directories present: {', '.join(unexpected)}")
        if missing:
            problems.append(f"declared top-level 3rdparty directories missing: {', '.join(missing)}")

    for entry in policy.get("project_owned_replacements", []):
        rel_path = Path(entry["path"])
        abs_path = ROOT / rel_path
        if not abs_path.exists():
            problems.append(f"project-owned replacement path missing: {rel_path.as_posix()}")
        else:
            notes.append(f"replacement ok: {entry['name']} ({rel_path.as_posix()})")

    for entry in policy.get("external_build_only", []):
        for rel in entry.get("repo_paths_must_be_absent", []):
            abs_path = ROOT / rel
            if abs_path.exists():
                problems.append(f"external-only dependency leaked into repository: {rel}")
        if entry["name"].lower() not in combined_docs_lower:
            problems.append(f"documentation is missing external dependency note: {entry['name']}")
        notes.append(f"external-only ok: {entry['name']}")

    for rel in policy.get("public_release_forbidden_paths", []):
        abs_path = ROOT / rel
        if abs_path.exists() and path_is_git_tracked(abs_path):
            problems.append(f"forbidden public-release path exists in repository: {rel}")

    if release_mode == "source-and-dynamic-binaries":
        required_phrases = [
            "dynamic-linked binary package",
            "staged vendor libraries are not published",
            "must be present on the target",
            "not part of the public source archive",
        ]
    else:
        required_phrases = [
            "source-only",
            "does not publish them as release assets",
            "not part of the public source archive",
            "do not publish dependency bundles",
        ]
    for phrase in required_phrases:
        if phrase not in combined_docs_lower:
            problems.append(f"required release-boundary phrase missing from docs: {phrase}")

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1

    print("[OK] dependency policy matches current repository and release posture")
    for note in notes:
        print(f"[INFO] {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
