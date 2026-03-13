#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Audit the binary redistribution policy against staged-library capture logic and docs."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "scripts/binary_release_policy.json"
STAGE_SCRIPT = ROOT / "scripts/stage_release_deps.sh"
README_FILES = [ROOT / "README.md"]
COMPLIANCE = ROOT / "docs/OPEN_SOURCE_COMPLIANCE.md"
MATRIX = ROOT / "docs/BINARY_REDISTRIBUTION_POLICY.md"
WORKFLOW = ROOT / ".github/workflows/release.yml"
ARTIFACT_WORKFLOW = ROOT / ".github/workflows/binary-artifact.yml"
ARTIFACT_DOC = ROOT / "docs/BINARY_ARTIFACT_WORKFLOW.md"
RELEASE_SCRIPT = ROOT / "scripts/create_release_archives.sh"

LIB_PATTERN = re.compile(r"\b[a-zA-Z0-9_.-]+\.(?:a|so(?:\.\d+)?)\b")
VALID_CLASSIFICATIONS = {"forbidden-by-default", "review-required", "public-release"}


def load_policy() -> dict:
    with POLICY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def staged_libs_from_script(text: str) -> set[str]:
    found = set(LIB_PATTERN.findall(text))
    return {name for name in found if name.startswith("lib") or name.startswith("libr")}


def main() -> int:
    problems: list[str] = []
    infos: list[str] = []

    policy = load_policy()
    stage_text = STAGE_SCRIPT.read_text(encoding="utf-8")
    readme_text = "\n".join(
        path.read_text(encoding="utf-8") for path in README_FILES if path.exists()
    )
    compliance_text = COMPLIANCE.read_text(encoding="utf-8")
    matrix_text = MATRIX.read_text(encoding="utf-8")
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    artifact_workflow_text = ARTIFACT_WORKFLOW.read_text(encoding="utf-8")
    artifact_doc_text = ARTIFACT_DOC.read_text(encoding="utf-8")

    default_public_binary_release = policy.get("default_public_binary_release")
    if not isinstance(default_public_binary_release, bool):
        problems.append("binary release policy must declare default_public_binary_release as a boolean")

    staged_entries = policy.get("staged_libraries", [])
    project_outputs = policy.get("project_outputs", [])
    allowed_release_artifacts = policy.get("allowed_release_artifacts", [])

    declared_staged = {entry["filename"] for entry in staged_entries}
    script_staged = staged_libs_from_script(stage_text)
    missing_from_policy = sorted(script_staged - declared_staged)
    if missing_from_policy:
        problems.append(
            "stage_release_deps.sh references libraries missing from binary policy: "
            + ", ".join(missing_from_policy)
        )

    public_project_outputs: list[str] = []

    for entry in staged_entries:
        filename = entry["filename"]
        classification = entry.get("classification")
        if classification not in VALID_CLASSIFICATIONS:
            problems.append(f"invalid classification for {filename}: {classification}")
        if entry.get("default_public_release") is not False:
            problems.append(f"staged external library must default to non-public release: {filename}")
        if filename not in matrix_text:
            problems.append(f"binary policy document is missing filename: {filename}")
        if not entry.get("reason"):
            problems.append(f"binary policy entry missing reason: {filename}")
        infos.append(f"policy ok: {filename} ({classification})")

    for entry in project_outputs:
        filename = entry["filename"]
        classification = entry.get("classification")
        public_flag = entry.get("default_public_release")
        if classification not in VALID_CLASSIFICATIONS:
            problems.append(f"invalid classification for {filename}: {classification}")
        if not isinstance(public_flag, bool):
            problems.append(f"project output must declare default_public_release as a boolean: {filename}")
        if public_flag and classification != "public-release":
            problems.append(f"public project output must use classification public-release: {filename}")
        if filename not in matrix_text:
            problems.append(f"binary policy document is missing filename: {filename}")
        if not entry.get("reason"):
            problems.append(f"binary policy entry missing reason: {filename}")
        if public_flag:
            public_project_outputs.append(filename)
        infos.append(f"policy ok: {filename} ({classification})")

    required_project_outputs = {
        "_visiong.so",
        "visiong.py",
        "libvisiong.so",
        "libvisiong.a",
        "visiong_cpp.zip",
        "visiong_python.zip",
    }
    declared_outputs = {entry["filename"] for entry in project_outputs}
    missing_outputs = sorted(required_project_outputs - declared_outputs)
    if missing_outputs:
        problems.append("binary policy is missing project outputs: " + ", ".join(missing_outputs))

    combined_doc_text = "\n".join([readme_text, compliance_text, matrix_text]).lower()
    if default_public_binary_release:
        required_doc_phrases = [
            "visiong binary packages",
            "staged vendor libraries are not published",
            "must be present on the target",
        ]
    else:
        required_doc_phrases = ["source-only", "do not publish", "vendor-linked binaries"]
    for phrase in required_doc_phrases:
        if phrase not in combined_doc_text:
            problems.append(f"binary release boundary phrase missing from docs: {phrase}")

    if not ARTIFACT_WORKFLOW.is_file():
        problems.append("binary artifact workflow is missing: .github/workflows/binary-artifact.yml")
    if not ARTIFACT_DOC.is_file():
        problems.append("binary artifact workflow doc is missing: docs/BINARY_ARTIFACT_WORKFLOW.md")
    if not RELEASE_SCRIPT.is_file():
        problems.append("release archive script is missing: scripts/create_release_archives.sh")

    if default_public_binary_release:
        if "create_release_archives.sh" not in workflow_text:
            problems.append("public binary release workflow must call create_release_archives.sh")
        if "visiong_cpp.zip" not in workflow_text or "visiong_python.zip" not in workflow_text:
            problems.append("release workflow must publish visiong_cpp.zip and visiong_python.zip")
        if not public_project_outputs:
            problems.append("binary release policy enables public binary release but no public project outputs are declared")

    if "softprops/action-gh-release" in artifact_workflow_text:
        problems.append("binary artifact workflow must not publish a GitHub release")
    if "actions/upload-artifact@v4" not in artifact_workflow_text:
        problems.append("binary artifact workflow must upload Actions artifacts")
    if "workflow_dispatch:" not in artifact_workflow_text:
        problems.append("binary artifact workflow must remain manual-only (workflow_dispatch)")
    if "create_release_archives.sh" not in artifact_workflow_text:
        problems.append("binary artifact workflow must call create_release_archives.sh")
    if "workflow artifacts only" not in artifact_doc_text.lower():
        problems.append("binary artifact workflow doc must state that outputs are workflow artifacts only")

    readme_full_text = readme_text
    for artifact in allowed_release_artifacts:
        if artifact not in matrix_text and artifact not in workflow_text and artifact not in readme_full_text:
            problems.append(f"allowed release artifact not referenced by docs/workflow: {artifact}")

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1

    print("[OK] binary release policy matches current release posture")
    for info in infos:
        print(f"[INFO] {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
