#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Audit whether first-party source comments are bilingual (English + Chinese).
审计一方源码注释是否为中英双语。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
SOURCE_EXTENSIONS = {".h", ".hpp", ".c", ".cc", ".cpp", ".cxx", ".py", ".cmake", ".in"}
SKIP_DIRS = {"3rdparty", "licenses", ".git", "build", "dist"}
SPECIAL_NAMES = {"CMakeLists.txt", "build.sh"}


@dataclass
class CommentEntry:
    path: Path
    line_no: int
    marker: str
    content: str


@dataclass
class AuditIssue:
    path: Path
    line_no: int
    content: str


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def looks_like_separator(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and set(stripped) <= {"=", "-", "_", "*", "/", "#"}


def should_scan(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    return path.suffix.lower() in SOURCE_EXTENSIONS or path.name in SPECIAL_NAMES or path.parts[0] == "scripts"


def iter_comment_entries(path: Path) -> Iterable[CommentEntry]:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_block = False

    for line_no, line in enumerate(lines, 1):
        stripped = line.lstrip()
        marker = None
        content = None

        if in_block:
            marker = "*"
            if "*/" in stripped:
                before = stripped.split("*/", 1)[0]
                if before.startswith("*"):
                    before = before[1:]
                content = before.strip()
                in_block = False
            else:
                content = stripped[1:].strip() if stripped.startswith("*") else stripped.strip()
        else:
            if stripped.startswith("//"):
                marker = "//"
                content = stripped[2:].strip()
            elif stripped.startswith("/*"):
                marker = "/*"
                after = stripped[2:]
                if "*/" in after:
                    content = after.split("*/", 1)[0].strip()
                else:
                    content = after.strip()
                    in_block = True
            elif path.suffix in {".sh", ".py", ".cmake", ".in"} or path.name in SPECIAL_NAMES or path.parts[0] == "scripts":
                if (
                    stripped.startswith("#")
                    and not stripped.startswith("#!")
                    and not stripped.startswith("#include")
                    and not stripped.startswith("#ifndef")
                    and not stripped.startswith("#define")
                    and not stripped.startswith("#endif")
                    and not stripped.startswith("#if")
                    and not stripped.startswith("#elif")
                    and not stripped.startswith("#else")
                ):
                    marker = "#"
                    content = stripped[1:].strip()

        if marker and content and not content.startswith("SPDX-License-Identifier"):
            yield CommentEntry(path=path, line_no=line_no, marker=marker, content=content)


def audit() -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    paths = sorted(path for path in ROOT.rglob("*") if path.is_file() and should_scan(path))

    for path in paths:
        entries = list(iter_comment_entries(path))
        by_line = {entry.line_no: entry for entry in entries}
        for entry in entries:
            if not entry.content or looks_like_separator(entry.content):
                continue
            if has_cjk(entry.content):
                continue
            prev_entry = by_line.get(entry.line_no - 1)
            next_entry = by_line.get(entry.line_no + 1)
            if (prev_entry and has_cjk(prev_entry.content)) or (next_entry and has_cjk(next_entry.content)):
                continue
            issues.append(AuditIssue(path=path, line_no=entry.line_no, content=entry.content))

    return issues


def main() -> int:
    issues = audit()
    if issues:
        print("Found comments without adjacent Chinese translation / 发现缺少相邻中文翻译的注释：")
        for issue in issues:
            rel = issue.path.relative_to(ROOT)
            print(f"- {rel}:{issue.line_no}: {issue.content}")
        return 1

    print("All audited first-party comments are bilingual / 所审计的一方注释均为中英双语。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
